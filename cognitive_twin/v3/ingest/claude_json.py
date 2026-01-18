"""
Claude JSON Data Parser

Parses conversation data from Claude JSON exports:
- data/Claude Data/conversations.json
- data/Claude Data/memories.json
- data/Claude Data/projects.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    ExtractorConfig,
    ExtractionResult,
    SourceProvider,
    TurnMetadata,
    TurnRole,
    UnifiedConversation,
    UnifiedTurn,
)

logger = logging.getLogger(__name__)


class ClaudeJSONParser:
    """
    Parse conversations from Claude JSON export files.
    
    Claude exports typically contain:
    - conversations.json: Main conversation data
    - memories.json: Persistent memories
    - projects.json: Project contexts
    - users.json: User information
    """
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        
    def extract_from_directory(self, data_dir: Path) -> ExtractionResult:
        """
        Extract all conversations from a Claude data directory.
        
        Args:
            data_dir: Path to Claude Data directory
            
        Returns:
            ExtractionResult with extracted conversations
        """
        started_at = datetime.now()
        conversations: List[UnifiedConversation] = []
        errors: List[str] = []
        total_found = 0
        total_skipped = 0
        
        # Load conversations.json
        conversations_file = data_dir / "conversations.json"
        if conversations_file.exists():
            try:
                convos, found, skipped = self._parse_conversations_file(conversations_file)
                conversations.extend(convos)
                total_found += found
                total_skipped += skipped
                logger.info(f"Extracted {len(convos)} conversations from conversations.json")
            except Exception as e:
                errors.append(f"Failed to parse conversations.json: {e}")
                logger.error(f"conversations.json parsing failed: {e}")
        else:
            logger.warning(f"conversations.json not found at {conversations_file}")
        
        # Load memories for context enrichment
        memories = {}
        memories_file = data_dir / "memories.json"
        if memories_file.exists():
            try:
                memories = self._load_memories(memories_file)
                logger.info(f"Loaded {len(memories)} memories for context")
            except Exception as e:
                logger.debug(f"Failed to load memories: {e}")
        
        # Load projects for context enrichment
        projects = {}
        projects_file = data_dir / "projects.json"
        if projects_file.exists():
            try:
                projects = self._load_projects(projects_file)
                logger.info(f"Loaded {len(projects)} projects for context")
            except Exception as e:
                logger.debug(f"Failed to load projects: {e}")
        
        # Enrich conversations with context
        for conv in conversations:
            self._enrich_with_context(conv, memories, projects)
        
        return ExtractionResult(
            conversations=conversations,
            source=SourceProvider.CLAUDE,
            total_found=total_found,
            total_extracted=len(conversations),
            total_skipped=total_skipped,
            errors=errors,
            started_at=started_at,
            completed_at=datetime.now(),
        )
    
    def _parse_conversations_file(
        self, 
        path: Path
    ) -> Tuple[List[UnifiedConversation], int, int]:
        """Parse the main conversations.json file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        conversations: List[UnifiedConversation] = []
        total_found = 0
        total_skipped = 0
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("conversations", data.get("items", [data]))
        else:
            return [], 0, 0
        
        for item in items:
            total_found += 1
            try:
                conv = self._parse_conversation(item)
                if conv and self._passes_filter(conv):
                    conversations.append(conv)
                else:
                    total_skipped += 1
            except Exception as e:
                logger.debug(f"Failed to parse conversation: {e}")
                total_skipped += 1
        
        return conversations, total_found, total_skipped
    
    def _parse_conversation(self, data: Dict[str, Any]) -> Optional[UnifiedConversation]:
        """Parse a single conversation from JSON data."""
        # Extract conversation ID
        conv_id = (
            data.get("uuid") or 
            data.get("id") or 
            data.get("conversation_id") or
            f"claude-{hash(str(data))}"
        )
        
        # Extract messages/turns
        messages = (
            data.get("chat_messages") or
            data.get("messages") or
            data.get("turns") or
            []
        )
        
        if not messages:
            return None
        
        turns: List[UnifiedTurn] = []
        
        for msg in messages:
            role_str = msg.get("sender", msg.get("role", "user"))
            role = self._map_role(role_str)
            
            # Skip system turns if configured
            if role == TurnRole.SYSTEM and not self.config.include_system_turns:
                continue
            
            # Extract content
            content = self._extract_content(msg)
            if not content:
                continue
            
            # Parse timestamp
            timestamp = None
            ts_str = msg.get("created_at") or msg.get("timestamp")
            if ts_str:
                try:
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass
            
            # Build metadata
            metadata = TurnMetadata(
                model_used=msg.get("model"),
                raw_metadata={
                    "index": msg.get("index"),
                    "feedback": msg.get("feedback"),
                },
            )
            
            turn = UnifiedTurn(
                role=role,
                content=content,
                source=SourceProvider.CLAUDE,
                original_id=msg.get("uuid"),
                timestamp=timestamp,
                metadata=metadata,
            )
            turns.append(turn)
        
        if len(turns) < self.config.min_turns:
            return None
        
        # Extract conversation-level metadata
        created_at = None
        created_str = data.get("created_at")
        if created_str:
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        
        return UnifiedConversation(
            conversation_id=f"claude-{conv_id}",
            source_provider=SourceProvider.CLAUDE,
            turns=turns,
            project_path=data.get("project", {}).get("path") if isinstance(data.get("project"), dict) else None,
            project_context=data.get("project", {}).get("name") if isinstance(data.get("project"), dict) else None,
            created_at=created_at,
            updated_at=None,
            raw_source=data,
        )
    
    def _extract_content(self, msg: Dict[str, Any]) -> str:
        """Extract text content from a message."""
        # Direct text field
        if isinstance(msg.get("text"), str):
            return msg["text"]
        
        # Content field (various formats)
        content = msg.get("content")
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            # Handle content blocks format
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block:
                        text_parts.append(block["text"])
            return "\n".join(text_parts)
        
        return ""
    
    def _map_role(self, role_str: str) -> TurnRole:
        """Map role string to TurnRole enum."""
        role_map = {
            "human": TurnRole.USER,
            "user": TurnRole.USER,
            "assistant": TurnRole.ASSISTANT,
            "ai": TurnRole.ASSISTANT,
            "system": TurnRole.SYSTEM,
            "tool": TurnRole.TOOL,
        }
        return role_map.get(role_str.lower(), TurnRole.USER)
    
    def _load_memories(self, path: Path) -> Dict[str, Any]:
        """Load memories file for context enrichment."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        memories = {}
        items = data if isinstance(data, list) else data.get("memories", [])
        
        for item in items:
            if isinstance(item, dict):
                mem_id = item.get("id") or item.get("uuid")
                if mem_id:
                    memories[mem_id] = item
        
        return memories
    
    def _load_projects(self, path: Path) -> Dict[str, Any]:
        """Load projects file for context enrichment."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        projects = {}
        items = data if isinstance(data, list) else data.get("projects", [])
        
        for item in items:
            if isinstance(item, dict):
                proj_id = item.get("id") or item.get("uuid")
                if proj_id:
                    projects[proj_id] = item
        
        return projects
    
    def _enrich_with_context(
        self, 
        conv: UnifiedConversation, 
        memories: Dict[str, Any],
        projects: Dict[str, Any],
    ):
        """Enrich conversation with memory and project context."""
        # This would add relevant memories/projects as context
        # Implementation depends on the specific enrichment needed
        pass
    
    def _passes_filter(self, conversation: UnifiedConversation) -> bool:
        """Check if conversation passes all filters."""
        if len(conversation.turns) < self.config.min_turns:
            return False
        if len(conversation.turns) > self.config.max_turns:
            return False
        
        total_length = sum(t.content_length for t in conversation.turns)
        if total_length < self.config.min_content_length:
            return False
        
        return True


def parse_claude_json(
    data_dir: Path,
    min_turns: int = 2,
) -> ExtractionResult:
    """
    Parse Claude JSON data with minimal configuration.
    
    Args:
        data_dir: Path to Claude Data directory
        min_turns: Minimum turns to include a conversation
        
    Returns:
        ExtractionResult with extracted conversations
    """
    config = ExtractorConfig(min_turns=min_turns)
    parser = ClaudeJSONParser(config)
    return parser.extract_from_directory(data_dir)

