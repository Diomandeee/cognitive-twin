"""
Prompt Logger Data Extractor

Extracts conversation data from the Prompt Logger system:
- ~/.claude/prompt-logs/verbose-all.jsonl (unified logs)
- ~/.claude/prompt-logs/prompts-all.jsonl (all prompts)
- ~/.claude/prompt-logs/sessions/*/verbose.jsonl (session logs)
- ~/.claude/plans/*.md (Claude plans as context)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from .types import (
    ExtractorConfig,
    ExtractionResult,
    GitContext,
    SourceProvider,
    ToolCall,
    TurnMetadata,
    TurnRole,
    UnifiedConversation,
    UnifiedTurn,
)

logger = logging.getLogger(__name__)


class PromptLoggerExtractor:
    """
    Extract conversations from Prompt Logger data sources.
    
    The Prompt Logger stores conversation data in multiple formats:
    1. verbose-all.jsonl: Unified log with full turn details
    2. prompts-all.jsonl: Simplified prompt records
    3. sessions/*/verbose.jsonl: Per-session verbose logs
    4. plans/*.md: Claude plans as context documents
    """
    
    DEFAULT_BASE_PATH = Path.home() / ".claude" / "prompt-logs"
    DEFAULT_PLANS_PATH = Path.home() / ".claude" / "plans"
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self.base_path = self.config.base_path or self.DEFAULT_BASE_PATH
        self.plans_path = self.DEFAULT_PLANS_PATH
        
    def extract_all(self) -> ExtractionResult:
        """
        Extract all conversations from Prompt Logger sources.
        
        Returns:
            ExtractionResult with all extracted conversations
        """
        started_at = datetime.now()
        conversations: List[UnifiedConversation] = []
        errors: List[str] = []
        total_found = 0
        total_skipped = 0
        
        # Extract from verbose-all.jsonl
        try:
            verbose_convos, verbose_found, verbose_skipped = self._extract_from_verbose_all()
            conversations.extend(verbose_convos)
            total_found += verbose_found
            total_skipped += verbose_skipped
            logger.info(f"Extracted {len(verbose_convos)} conversations from verbose-all.jsonl")
        except Exception as e:
            errors.append(f"Failed to extract from verbose-all.jsonl: {e}")
            logger.error(f"verbose-all.jsonl extraction failed: {e}")
        
        # Extract from session directories
        try:
            session_convos, session_found, session_skipped = self._extract_from_sessions()
            conversations.extend(session_convos)
            total_found += session_found
            total_skipped += session_skipped
            logger.info(f"Extracted {len(session_convos)} conversations from sessions")
        except Exception as e:
            errors.append(f"Failed to extract from sessions: {e}")
            logger.error(f"Session extraction failed: {e}")
        
        # Extract plans as context documents
        try:
            plan_contexts = self._extract_plans()
            logger.info(f"Extracted {len(plan_contexts)} plans as context")
        except Exception as e:
            errors.append(f"Failed to extract plans: {e}")
            logger.error(f"Plan extraction failed: {e}")
        
        return ExtractionResult(
            conversations=conversations,
            source=SourceProvider.PROMPT_LOGGER,
            total_found=total_found,
            total_extracted=len(conversations),
            total_skipped=total_skipped,
            errors=errors,
            started_at=started_at,
            completed_at=datetime.now(),
        )
    
    def _extract_from_verbose_all(self) -> Tuple[List[UnifiedConversation], int, int]:
        """Extract conversations from verbose-all.jsonl."""
        verbose_path = self.base_path / "verbose-all.jsonl"
        if not verbose_path.exists():
            logger.warning(f"verbose-all.jsonl not found at {verbose_path}")
            return [], 0, 0
        
        conversations: List[UnifiedConversation] = []
        session_turns: Dict[str, List[Dict[str, Any]]] = {}
        total_found = 0
        total_skipped = 0
        
        # Read all entries and group by session
        for entry in self._read_jsonl(verbose_path):
            total_found += 1
            session_id = entry.get("session_id", "unknown")
            
            if session_id not in session_turns:
                session_turns[session_id] = []
            session_turns[session_id].append(entry)
        
        # Convert session groups to conversations
        for session_id, entries in session_turns.items():
            try:
                conversation = self._entries_to_conversation(session_id, entries)
                if conversation and self._passes_filter(conversation):
                    conversations.append(conversation)
                else:
                    total_skipped += 1
            except Exception as e:
                logger.debug(f"Failed to convert session {session_id}: {e}")
                total_skipped += 1
        
        return conversations, total_found, total_skipped
    
    def _extract_from_sessions(self) -> Tuple[List[UnifiedConversation], int, int]:
        """Extract conversations from individual session directories."""
        sessions_path = self.base_path / "sessions"
        if not sessions_path.exists():
            logger.warning(f"Sessions directory not found at {sessions_path}")
            return [], 0, 0
        
        conversations: List[UnifiedConversation] = []
        total_found = 0
        total_skipped = 0
        
        for session_dir in sessions_path.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_id = session_dir.name
            verbose_file = session_dir / "verbose.jsonl"
            
            if not verbose_file.exists():
                continue
            
            entries = list(self._read_jsonl(verbose_file))
            total_found += len(entries)
            
            if entries:
                try:
                    conversation = self._entries_to_conversation(session_id, entries)
                    if conversation and self._passes_filter(conversation):
                        conversations.append(conversation)
                    else:
                        total_skipped += 1
                except Exception as e:
                    logger.debug(f"Failed to convert session {session_id}: {e}")
                    total_skipped += 1
        
        return conversations, total_found, total_skipped
    
    def _extract_plans(self) -> List[Dict[str, Any]]:
        """Extract Claude plans as context documents."""
        if not self.plans_path.exists():
            return []
        
        plans = []
        for plan_file in self.plans_path.glob("*.md"):
            try:
                content = plan_file.read_text(encoding="utf-8")
                plans.append({
                    "name": plan_file.stem,
                    "path": str(plan_file),
                    "content": content,
                    "type": "plan",
                })
            except Exception as e:
                logger.debug(f"Failed to read plan {plan_file}: {e}")
        
        return plans
    
    def _entries_to_conversation(
        self, 
        session_id: str, 
        entries: List[Dict[str, Any]]
    ) -> Optional[UnifiedConversation]:
        """Convert a list of prompt logger entries to a unified conversation."""
        if not entries:
            return None
        
        # Sort by timestamp
        entries = sorted(entries, key=lambda e: e.get("timestamp", ""))
        
        turns: List[UnifiedTurn] = []
        
        for entry in entries:
            # Determine source provider
            source_str = entry.get("source", "unknown")
            source = self._map_source(source_str)
            
            # Skip if source is filtered
            if not self._source_allowed(source):
                continue
            
            # Extract user turn from prompt_text
            prompt_text = entry.get("prompt_text", "")
            if prompt_text:
                user_turn = self._create_turn(
                    role=TurnRole.USER,
                    content=prompt_text,
                    source=source,
                    entry=entry,
                )
                turns.append(user_turn)
            
            # Extract assistant turns
            assistant_turns = entry.get("assistant_turns", [])
            if isinstance(assistant_turns, list):
                for asst_turn in assistant_turns:
                    if isinstance(asst_turn, dict):
                        content = asst_turn.get("content", "")
                    elif isinstance(asst_turn, str):
                        content = asst_turn
                    else:
                        continue
                    
                    if content:
                        turn = self._create_turn(
                            role=TurnRole.ASSISTANT,
                            content=content,
                            source=source,
                            entry=entry,
                        )
                        turns.append(turn)
        
        if len(turns) < self.config.min_turns:
            return None
        
        # Get metadata from first entry
        first_entry = entries[0]
        git_context_json = first_entry.get("git_context_json")
        git_context = None
        if git_context_json:
            try:
                if isinstance(git_context_json, str):
                    git_data = json.loads(git_context_json)
                else:
                    git_data = git_context_json
                git_context = GitContext(
                    repo_name=git_data.get("repo_name"),
                    branch=git_data.get("branch"),
                    commit_sha=git_data.get("commit_sha"),
                    is_dirty=git_data.get("is_dirty", False),
                )
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Parse timestamp
        created_at = None
        timestamp_str = first_entry.get("timestamp")
        if timestamp_str:
            try:
                created_at = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        
        return UnifiedConversation(
            conversation_id=f"pl-{session_id}",
            source_provider=self._map_source(first_entry.get("source", "unknown")),
            turns=turns,
            project_path=first_entry.get("cwd"),
            session_id=session_id,
            created_at=created_at,
        )
    
    def _create_turn(
        self,
        role: TurnRole,
        content: str,
        source: SourceProvider,
        entry: Dict[str, Any],
    ) -> UnifiedTurn:
        """Create a unified turn from entry data."""
        # Parse timestamp
        timestamp = None
        ts_str = entry.get("timestamp") or entry.get("prompt_received_at")
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        
        # Extract tool calls
        tool_calls: List[ToolCall] = []
        tool_call_count = entry.get("tool_calls", 0)
        # Note: Detailed tool call data may not be in all entries
        
        # Build metadata
        git_context = None
        git_json = entry.get("git_context_json")
        if git_json:
            try:
                if isinstance(git_json, str):
                    git_data = json.loads(git_json)
                else:
                    git_data = git_json
                git_context = GitContext(
                    repo_name=git_data.get("repo_name"),
                    branch=git_data.get("branch"),
                    commit_sha=git_data.get("commit_sha"),
                    is_dirty=git_data.get("is_dirty", False),
                )
            except (json.JSONDecodeError, TypeError):
                pass
        
        metadata = TurnMetadata(
            cwd=entry.get("cwd"),
            git_context=git_context,
            file_diffs=entry.get("file_diffs", 0),
            affected_files=entry.get("affected_targets", []),
            raw_metadata={
                "prompt_id": entry.get("prompt_id"),
                "parent_prompt_id": entry.get("parent_prompt_id"),
            },
        )
        
        return UnifiedTurn(
            role=role,
            content=content,
            source=source,
            original_id=entry.get("prompt_id"),
            timestamp=timestamp,
            tool_calls=tool_calls,
            metadata=metadata,
        )
    
    def _map_source(self, source_str: str) -> SourceProvider:
        """Map source string to SourceProvider enum."""
        source_map = {
            "claude": SourceProvider.CLAUDE,
            "codex": SourceProvider.CODEX,
            "cursor": SourceProvider.CURSOR,
            "openai": SourceProvider.OPENAI,
        }
        return source_map.get(source_str.lower(), SourceProvider.UNKNOWN)
    
    def _source_allowed(self, source: SourceProvider) -> bool:
        """Check if source passes filter configuration."""
        if self.config.include_sources:
            return source in self.config.include_sources
        if self.config.exclude_sources:
            return source not in self.config.exclude_sources
        return True
    
    def _passes_filter(self, conversation: UnifiedConversation) -> bool:
        """Check if conversation passes all filters."""
        # Turn count filter
        if len(conversation.turns) < self.config.min_turns:
            return False
        if len(conversation.turns) > self.config.max_turns:
            return False
        
        # Content length filter
        total_length = sum(t.content_length for t in conversation.turns)
        if total_length < self.config.min_content_length:
            return False
        
        # Date filter
        if self.config.after_date and conversation.created_at:
            if conversation.created_at < self.config.after_date:
                return False
        if self.config.before_date and conversation.created_at:
            if conversation.created_at > self.config.before_date:
                return False
        
        return True
    
    def _read_jsonl(self, path: Path) -> Generator[Dict[str, Any], None, None]:
        """Read JSONL file line by line."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Invalid JSON at {path}:{line_num}: {e}")
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")


# Convenience function for direct extraction
def extract_prompt_logger_data(
    base_path: Optional[Path] = None,
    min_turns: int = 2,
    include_sources: Optional[List[str]] = None,
) -> ExtractionResult:
    """
    Extract all Prompt Logger data with minimal configuration.
    
    Args:
        base_path: Path to prompt-logs directory (defaults to ~/.claude/prompt-logs)
        min_turns: Minimum turns to include a conversation
        include_sources: List of sources to include (claude, codex, cursor)
    
    Returns:
        ExtractionResult with all extracted conversations
    """
    config = ExtractorConfig(
        base_path=base_path,
        min_turns=min_turns,
        include_sources=[SourceProvider(s) for s in include_sources] if include_sources else None,
    )
    extractor = PromptLoggerExtractor(config)
    return extractor.extract_all()

