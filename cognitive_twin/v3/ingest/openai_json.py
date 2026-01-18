"""
OpenAI JSON Data Parser

Parses conversation data from OpenAI JSON exports:
- data/Open AI Data/conversations.json
- data/Open AI Data/conversation_openai.json
- data/Open AI Data/conversation_new.json
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
    ToolCall,
    TurnMetadata,
    TurnRole,
    UnifiedConversation,
    UnifiedTurn,
)

logger = logging.getLogger(__name__)


class OpenAIJSONParser:
    """
    Parse conversations from OpenAI JSON export files.
    
    OpenAI exports typically contain conversation data with:
    - Nested mapping structure
    - Multiple message types (user, assistant, system, tool)
    - Tool call information
    """
    
    CONVERSATION_FILES = [
        "conversations.json",
        "conversation_openai.json",
        "conversation_new.json",
        "conversations copy.json",
    ]
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        
    def extract_from_directory(self, data_dir: Path) -> ExtractionResult:
        """
        Extract all conversations from an OpenAI data directory.
        
        Args:
            data_dir: Path to OpenAI Data directory
            
        Returns:
            ExtractionResult with extracted conversations
        """
        started_at = datetime.now()
        conversations: List[UnifiedConversation] = []
        errors: List[str] = []
        total_found = 0
        total_skipped = 0
        seen_ids = set()  # Track to avoid duplicates across files
        
        for filename in self.CONVERSATION_FILES:
            filepath = data_dir / filename
            if not filepath.exists():
                continue
            
            try:
                convos, found, skipped = self._parse_file(filepath, seen_ids)
                conversations.extend(convos)
                total_found += found
                total_skipped += skipped
                logger.info(f"Extracted {len(convos)} conversations from {filename}")
            except Exception as e:
                errors.append(f"Failed to parse {filename}: {e}")
                logger.error(f"{filename} parsing failed: {e}")
        
        return ExtractionResult(
            conversations=conversations,
            source=SourceProvider.OPENAI,
            total_found=total_found,
            total_extracted=len(conversations),
            total_skipped=total_skipped,
            errors=errors,
            started_at=started_at,
            completed_at=datetime.now(),
        )
    
    def _parse_file(
        self, 
        path: Path,
        seen_ids: set,
    ) -> Tuple[List[UnifiedConversation], int, int]:
        """Parse a single OpenAI JSON file."""
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
            
            # Get conversation ID
            conv_id = (
                item.get("id") or
                item.get("conversation_id") or
                item.get("uuid") or
                str(hash(json.dumps(item, sort_keys=True)))[:16]
            )
            
            # Skip if already processed
            if conv_id in seen_ids:
                total_skipped += 1
                continue
            
            try:
                conv = self._parse_conversation(item)
                if conv and self._passes_filter(conv):
                    conversations.append(conv)
                    seen_ids.add(conv_id)
                else:
                    total_skipped += 1
            except Exception as e:
                logger.debug(f"Failed to parse conversation {conv_id}: {e}")
                total_skipped += 1
        
        return conversations, total_found, total_skipped
    
    def _parse_conversation(self, data: Dict[str, Any]) -> Optional[UnifiedConversation]:
        """Parse a single OpenAI conversation."""
        conv_id = (
            data.get("id") or
            data.get("conversation_id") or
            f"openai-{hash(str(data))}"
        )
        
        # Extract messages from various possible structures
        turns = self._extract_turns(data)
        
        if len(turns) < self.config.min_turns:
            return None
        
        # Extract metadata
        created_at = None
        create_time = data.get("create_time")
        if create_time:
            try:
                if isinstance(create_time, (int, float)):
                    created_at = datetime.fromtimestamp(create_time)
                elif isinstance(create_time, str):
                    created_at = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
            except (ValueError, OSError):
                pass
        
        updated_at = None
        update_time = data.get("update_time")
        if update_time:
            try:
                if isinstance(update_time, (int, float)):
                    updated_at = datetime.fromtimestamp(update_time)
                elif isinstance(update_time, str):
                    updated_at = datetime.fromisoformat(update_time.replace("Z", "+00:00"))
            except (ValueError, OSError):
                pass
        
        return UnifiedConversation(
            conversation_id=f"openai-{conv_id}",
            source_provider=SourceProvider.OPENAI,
            turns=turns,
            project_context=data.get("title"),
            created_at=created_at,
            updated_at=updated_at,
            raw_source=data,
        )
    
    def _extract_turns(self, data: Dict[str, Any]) -> List[UnifiedTurn]:
        """Extract turns from OpenAI conversation data."""
        turns: List[UnifiedTurn] = []
        
        # Check for mapping structure (OpenAI export format)
        mapping = data.get("mapping", {})
        if mapping:
            turns = self._extract_from_mapping(mapping)
        else:
            # Check for simple messages array
            messages = data.get("messages", data.get("turns", []))
            if messages:
                turns = self._extract_from_messages(messages)
        
        return turns
    
    def _extract_from_mapping(self, mapping: Dict[str, Any]) -> List[UnifiedTurn]:
        """Extract turns from OpenAI mapping structure."""
        turns: List[UnifiedTurn] = []
        
        # Build ordered list of nodes
        nodes: List[Tuple[float, Dict[str, Any]]] = []
        
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            
            message = node.get("message")
            if not message or not isinstance(message, dict):
                continue
            
            # Get ordering timestamp
            create_time = message.get("create_time", 0)
            if create_time is None:
                create_time = 0
            
            nodes.append((create_time, message))
        
        # Sort by creation time
        nodes.sort(key=lambda x: x[0])
        
        for _, message in nodes:
            turn = self._parse_message(message)
            if turn:
                turns.append(turn)
        
        return turns
    
    def _extract_from_messages(self, messages: List[Dict[str, Any]]) -> List[UnifiedTurn]:
        """Extract turns from simple messages array."""
        turns: List[UnifiedTurn] = []
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            turn = self._parse_message(msg)
            if turn:
                turns.append(turn)
        
        return turns
    
    def _parse_message(self, message: Dict[str, Any]) -> Optional[UnifiedTurn]:
        """Parse a single message into a UnifiedTurn."""
        # Get author/role
        author = message.get("author", {})
        role_str = author.get("role") if isinstance(author, dict) else message.get("role", "user")
        
        if not role_str:
            return None
        
        role = self._map_role(role_str)
        
        # Skip system turns if configured
        if role == TurnRole.SYSTEM and not self.config.include_system_turns:
            return None
        
        # Skip tool results if configured
        if role == TurnRole.TOOL and not self.config.include_tool_results:
            return None
        
        # Extract content
        content = self._extract_content(message)
        if not content:
            return None
        
        # Parse timestamp
        timestamp = None
        create_time = message.get("create_time")
        if create_time:
            try:
                if isinstance(create_time, (int, float)):
                    timestamp = datetime.fromtimestamp(create_time)
            except (ValueError, OSError):
                pass
        
        # Extract tool calls
        tool_calls = self._extract_tool_calls(message)
        
        # Build metadata
        metadata = TurnMetadata(
            model_used=message.get("metadata", {}).get("model_slug"),
            raw_metadata={
                "message_id": message.get("id"),
                "weight": message.get("weight"),
                "end_turn": message.get("end_turn"),
            },
        )
        
        return UnifiedTurn(
            role=role,
            content=content,
            source=SourceProvider.OPENAI,
            original_id=message.get("id"),
            timestamp=timestamp,
            tool_calls=tool_calls,
            metadata=metadata,
        )
    
    def _extract_content(self, message: Dict[str, Any]) -> str:
        """Extract text content from a message."""
        content = message.get("content")
        
        if not content:
            return ""
        
        # Handle content parts structure
        if isinstance(content, dict):
            parts = content.get("parts", [])
            if parts:
                text_parts = []
                for part in parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        if part.get("content_type") == "text":
                            text_parts.append(part.get("text", ""))
                return "\n".join(text_parts)
            
            # Direct text field
            if "text" in content:
                return content["text"]
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)
        
        return ""
    
    def _extract_tool_calls(self, message: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from a message."""
        tool_calls: List[ToolCall] = []
        
        # Check for tool_calls in message
        calls = message.get("tool_calls", [])
        if not calls:
            # Check in content
            content = message.get("content", {})
            if isinstance(content, dict):
                calls = content.get("tool_calls", [])
        
        for call in calls:
            if not isinstance(call, dict):
                continue
            
            function = call.get("function", {})
            
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            
            tool_calls.append(ToolCall(
                tool_id=call.get("id", ""),
                tool_name=function.get("name", "unknown"),
                arguments=arguments,
            ))
        
        return tool_calls
    
    def _map_role(self, role_str: str) -> TurnRole:
        """Map role string to TurnRole enum."""
        role_map = {
            "user": TurnRole.USER,
            "human": TurnRole.USER,
            "assistant": TurnRole.ASSISTANT,
            "system": TurnRole.SYSTEM,
            "tool": TurnRole.TOOL,
            "function": TurnRole.TOOL,
        }
        return role_map.get(role_str.lower(), TurnRole.USER)
    
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


def parse_openai_json(
    data_dir: Path,
    min_turns: int = 2,
) -> ExtractionResult:
    """
    Parse OpenAI JSON data with minimal configuration.
    
    Args:
        data_dir: Path to OpenAI Data directory
        min_turns: Minimum turns to include a conversation
        
    Returns:
        ExtractionResult with extracted conversations
    """
    config = ExtractorConfig(min_turns=min_turns)
    parser = OpenAIJSONParser(config)
    return parser.extract_from_directory(data_dir)

