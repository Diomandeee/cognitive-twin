"""
Supabase Extractor - Extract 100K+ turns from the Supabase memory_turns table.

This is the primary source of training data - real user interactions
that have been logged over time.
"""

import os
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of Supabase extraction."""
    conversations: List[Any]
    total_turns: int
    total_conversations: int
    extraction_time: float


class SupabaseExtractor:
    """
    Extract conversations from Supabase memory_turns table.
    
    The memory_turns table contains:
    - 100K+ turns from real user interactions
    - Organized by session_id and project_id
    - Rich metadata for filtering
    """
    
    def __init__(
        self, 
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        batch_size: int = 1000,
    ):
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        self.batch_size = batch_size
        self._client = None
        
    @property
    def client(self):
        """Lazy initialization of Supabase client."""
        if self._client is None:
            try:
                from supabase import create_client, Client
                if not self.supabase_url or not self.supabase_key:
                    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required")
                self._client = create_client(self.supabase_url, self.supabase_key)
            except ImportError:
                raise ImportError("supabase package not installed: pip install supabase")
        return self._client
    
    def extract_all_turns(
        self, 
        limit: Optional[int] = None,
        min_content_length: int = 20,
    ) -> ExtractionResult:
        """
        Extract all turns from memory_turns table.
        
        Args:
            limit: Optional limit on number of turns
            min_content_length: Minimum content length to include
            
        Returns:
            ExtractionResult with organized conversations
        """
        import time
        start_time = time.time()
        
        all_turns = []
        offset = 0
        
        logger.info("Extracting turns from Supabase memory_turns...")
        
        while True:
            query = self.client.table("memory_turns").select("*")
            query = query.order("created_at", desc=False)
            query = query.range(offset, offset + self.batch_size - 1)
            
            result = query.execute()
            
            if not result.data:
                break
                
            all_turns.extend(result.data)
            offset += self.batch_size
            
            logger.info(f"  Extracted {len(all_turns)} turns...")
            
            if limit and len(all_turns) >= limit:
                all_turns = all_turns[:limit]
                break
        
        # Organize into conversations by session_id
        conversations = self._organize_into_conversations(all_turns)
        
        extraction_time = time.time() - start_time
        
        logger.info(f"Extracted {len(all_turns)} turns, {len(conversations)} conversations in {extraction_time:.1f}s")
        
        return ExtractionResult(
            conversations=conversations,
            total_turns=len(all_turns),
            total_conversations=len(conversations),
            extraction_time=extraction_time,
        )
    
    def extract_by_project(
        self, 
        project_id: str,
        limit: Optional[int] = None,
    ) -> ExtractionResult:
        """Extract turns for a specific project."""
        import time
        start_time = time.time()
        
        query = self.client.table("memory_turns").select("*")
        query = query.eq("project_id", project_id)
        query = query.order("created_at", desc=False)
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        turns = result.data or []
        
        conversations = self._organize_into_conversations(turns)
        
        return ExtractionResult(
            conversations=conversations,
            total_turns=len(turns),
            total_conversations=len(conversations),
            extraction_time=time.time() - start_time,
        )
    
    def extract_recent(
        self, 
        days: int = 30,
        limit: Optional[int] = None,
    ) -> ExtractionResult:
        """Extract recent turns from the last N days."""
        import time
        from datetime import timedelta
        
        start_time = time.time()
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = self.client.table("memory_turns").select("*")
        query = query.gte("created_at", cutoff.isoformat())
        query = query.order("created_at", desc=False)
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        turns = result.data or []
        
        conversations = self._organize_into_conversations(turns)
        
        return ExtractionResult(
            conversations=conversations,
            total_turns=len(turns),
            total_conversations=len(conversations),
            extraction_time=time.time() - start_time,
        )
    
    def _organize_into_conversations(
        self, 
        turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Organize flat turns into conversations grouped by session_id.
        
        Args:
            turns: List of turn dictionaries
            
        Returns:
            List of conversation dictionaries
        """
        from collections import defaultdict
        
        # Group by session_id
        sessions = defaultdict(list)
        for turn in turns:
            session_id = turn.get("session_id") or turn.get("conversation_id") or "unknown"
            sessions[session_id].append(turn)
        
        conversations = []
        for session_id, session_turns in sessions.items():
            # Sort by timestamp
            session_turns.sort(key=lambda t: t.get("created_at", ""))
            
            # Build conversation structure
            conv_turns = []
            for turn in session_turns:
                role = turn.get("role", "user")
                # Check both content_text (Supabase schema) and content (fallback)
                content = turn.get("content_text") or turn.get("content", "")
                
                if not content or len(content) < 10:
                    continue
                
                conv_turns.append({
                    "role": role,
                    "content": content,
                    "timestamp": turn.get("created_at"),
                    "metadata": {
                        "turn_id": turn.get("id"),
                        "project_id": turn.get("project_id"),
                        "source": turn.get("source", "supabase"),
                        "phase": turn.get("phase"),
                        "salience_score": turn.get("salience_score"),
                    }
                })
            
            if len(conv_turns) >= 2:  # At least user + assistant
                conversations.append({
                    "conversation_id": session_id,
                    "turns": conv_turns,
                    "source": "supabase",
                    "metadata": {
                        "turn_count": len(conv_turns),
                        "first_timestamp": session_turns[0].get("created_at"),
                        "last_timestamp": session_turns[-1].get("created_at"),
                    }
                })
        
        return conversations
    
    def get_turn_count(self) -> int:
        """Get total number of turns in the database."""
        result = self.client.table("memory_turns").select("id", count="exact").execute()
        return result.count or 0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions in the database."""
        # Get count by session
        result = self.client.rpc("get_session_stats").execute()
        return result.data if result.data else {}


class ContextBatcher:
    """
    Smart context batching for 400K context window.
    
    Accumulates conversations into batches that maximize
    context utilization while maintaining coherence.
    """
    
    def __init__(
        self,
        max_context_tokens: int = 400_000,
        max_output_tokens: int = 128_000,
        tokens_per_char: float = 0.25,  # Rough estimate
        target_utilization: float = 0.7,  # Use 70% of context
    ):
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.tokens_per_char = tokens_per_char
        self.target_utilization = target_utilization
        self.target_tokens = int(max_context_tokens * target_utilization)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.tokens_per_char)
    
    def create_batches(
        self, 
        conversations: List[Dict[str, Any]],
        include_global_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Create batches of conversations that fit within context limits.
        
        Each batch contains:
        - Global context (style signature, project info)
        - Related conversations (by topic/project)
        - Target conversation for generation
        
        Args:
            conversations: List of conversation dicts
            include_global_context: Whether to include global context
            
        Returns:
            List of batch dictionaries
        """
        batches = []
        
        # Sort conversations by project/topic for coherent batching
        sorted_convs = self._sort_by_coherence(conversations)
        
        # Global context that persists across batches
        global_context = self._build_global_context(conversations) if include_global_context else ""
        global_tokens = self.estimate_tokens(global_context)
        
        # Available tokens per batch
        available_per_batch = self.target_tokens - global_tokens - 10000  # Reserve for output
        
        current_batch = []
        current_tokens = 0
        
        for conv in sorted_convs:
            conv_text = self._serialize_conversation(conv)
            conv_tokens = self.estimate_tokens(conv_text)
            
            if current_tokens + conv_tokens > available_per_batch:
                # Finalize current batch
                if current_batch:
                    batches.append({
                        "conversations": current_batch,
                        "global_context": global_context,
                        "total_tokens": current_tokens + global_tokens,
                        "conversation_count": len(current_batch),
                    })
                
                # Start new batch
                current_batch = [conv]
                current_tokens = conv_tokens
            else:
                current_batch.append(conv)
                current_tokens += conv_tokens
        
        # Don't forget the last batch
        if current_batch:
            batches.append({
                "conversations": current_batch,
                "global_context": global_context,
                "total_tokens": current_tokens + global_tokens,
                "conversation_count": len(current_batch),
            })
        
        logger.info(f"Created {len(batches)} batches from {len(conversations)} conversations")
        logger.info(f"Average batch size: {sum(b['conversation_count'] for b in batches) / len(batches):.1f} conversations")
        
        return batches
    
    def _sort_by_coherence(
        self, 
        conversations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sort conversations for coherent batching (by project, topic, time)."""
        def sort_key(conv):
            metadata = conv.get("metadata", {})
            return (
                metadata.get("project_id", "zzz"),  # Group by project
                conv.get("source", "zzz"),  # Then by source
                metadata.get("first_timestamp", ""),  # Then by time
            )
        
        return sorted(conversations, key=sort_key)
    
    def _build_global_context(
        self, 
        conversations: List[Dict[str, Any]],
    ) -> str:
        """
        Build global context from conversation patterns.
        
        This is the "memory" that persists across batches:
        - Style signature
        - Common patterns
        - User preferences
        """
        # Extract common patterns
        projects = set()
        sources = set()
        turn_count = 0
        
        for conv in conversations[:100]:  # Sample
            metadata = conv.get("metadata", {})
            if metadata.get("project_id"):
                projects.add(metadata["project_id"])
            sources.add(conv.get("source", "unknown"))
            turn_count += len(conv.get("turns", []))
        
        context = f"""GLOBAL CONTEXT:
- Total conversations: {len(conversations)}
- Sample turn count: {turn_count}
- Projects: {len(projects)}
- Sources: {', '.join(sources)}

STYLE GUIDELINES:
- Respond directly without permission-seeking
- Execute tasks immediately when directives are clear
- Provide complete implementations without omissions
- Use numbered lists for multi-step processes
- Be concise but thorough
"""
        return context
    
    def _serialize_conversation(self, conv: Dict[str, Any]) -> str:
        """Serialize a conversation to text."""
        lines = [f"--- Conversation: {conv.get('conversation_id', 'unknown')} ---"]
        
        for turn in conv.get("turns", []):
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)


class GlobalContextBuilder:
    """
    Builds rich global context for batch generation.
    
    Extracts style patterns, domain-specific context, and few-shot examples
    from the training data to guide generation.
    """
    
    # Domain keywords for classification
    CODING_KEYWORDS = [
        "implement", "function", "class", "code", "debug", "error", "api",
        "python", "javascript", "typescript", "rust", "sql", "database",
        "algorithm", "refactor", "test", "deploy", "git", "docker"
    ]
    
    GENERAL_KEYWORDS = [
        "explain", "describe", "what is", "how does", "summarize", "analyze",
        "compare", "suggest", "recommend", "help me", "tell me"
    ]
    
    def __init__(self, max_examples_per_domain: int = 3):
        self.max_examples = max_examples_per_domain
        self._style_signature: Optional[str] = None
        self._domain_examples: Dict[str, List[Dict]] = {}
        self._user_preferences: Dict[str, Any] = {}
    
    def build_from_conversations(
        self, 
        conversations: List[Dict[str, Any]],
    ) -> str:
        """
        Build comprehensive global context from conversations.
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            Global context string for inclusion in batch prompts
        """
        # Extract patterns
        self._extract_style_signature(conversations)
        self._extract_domain_examples(conversations)
        self._extract_user_preferences(conversations)
        
        # Build context string
        context_parts = []
        
        # Style signature
        context_parts.append("## STYLE SIGNATURE")
        context_parts.append(self._style_signature or self._default_style())
        
        # Domain-specific guidance
        context_parts.append("\n## DOMAIN CONTEXT")
        for domain, examples in self._domain_examples.items():
            if examples:
                context_parts.append(f"\n### {domain.upper()} Tasks")
                context_parts.append(self._format_examples(examples[:self.max_examples]))
        
        # User preferences
        if self._user_preferences:
            context_parts.append("\n## USER PREFERENCES")
            for pref, value in self._user_preferences.items():
                context_parts.append(f"- {pref}: {value}")
        
        return "\n".join(context_parts)
    
    def _extract_style_signature(self, conversations: List[Dict[str, Any]]) -> None:
        """Extract style patterns from assistant responses."""
        # Analyze response patterns
        uses_numbered_lists = 0
        uses_code_blocks = 0
        uses_headers = 0
        avg_response_length = 0
        total_responses = 0
        
        for conv in conversations[:200]:  # Sample
            for turn in conv.get("turns", []):
                if turn.get("role") == "assistant":
                    content = turn.get("content", "")
                    total_responses += 1
                    avg_response_length += len(content)
                    
                    if re.search(r'^\d+\.', content, re.MULTILINE):
                        uses_numbered_lists += 1
                    if '```' in content:
                        uses_code_blocks += 1
                    if re.search(r'^#{1,3}\s', content, re.MULTILINE):
                        uses_headers += 1
        
        if total_responses > 0:
            avg_response_length //= total_responses
            
            self._style_signature = f"""Preferred Response Style:
- Average response length: ~{avg_response_length} characters
- Use numbered lists: {uses_numbered_lists * 100 // total_responses}% of responses
- Use code blocks: {uses_code_blocks * 100 // total_responses}% of responses
- Use headers: {uses_headers * 100 // total_responses}% of responses

CRITICAL RULES:
- Never end with a question like "Would you like..." or "Should I..."
- Execute tasks directly without asking for confirmation
- Provide complete implementations without placeholders
- Be concise but thorough"""
    
    def _extract_domain_examples(self, conversations: List[Dict[str, Any]]) -> None:
        """Extract good examples for each domain."""
        self._domain_examples = {"coding": [], "general": [], "mixed": []}
        
        for conv in conversations:
            turns = conv.get("turns", [])
            if len(turns) < 2:
                continue
            
            # Look for user-assistant pairs
            for i in range(len(turns) - 1):
                if turns[i].get("role") == "user" and turns[i+1].get("role") == "assistant":
                    user_content = turns[i].get("content", "").lower()
                    assistant_content = turns[i+1].get("content", "")
                    
                    # Skip if assistant asks questions
                    if any(q in assistant_content.lower() for q in [
                        "would you like", "should i", "do you want", "can you confirm"
                    ]):
                        continue
                    
                    # Classify domain
                    domain = self._classify_domain(user_content)
                    
                    # Add example if high quality
                    if len(assistant_content) > 100:
                        example = {
                            "prompt": turns[i].get("content", ""),
                            "response": assistant_content,
                        }
                        if len(self._domain_examples[domain]) < self.max_examples * 2:
                            self._domain_examples[domain].append(example)
    
    def _classify_domain(self, text: str) -> str:
        """Classify text into a domain."""
        text_lower = text.lower()
        
        coding_score = sum(1 for kw in self.CODING_KEYWORDS if kw in text_lower)
        general_score = sum(1 for kw in self.GENERAL_KEYWORDS if kw in text_lower)
        
        if coding_score > general_score:
            return "coding"
        elif general_score > coding_score:
            return "general"
        else:
            return "mixed"
    
    def _extract_user_preferences(self, conversations: List[Dict[str, Any]]) -> None:
        """Extract user preferences from conversation patterns."""
        corrections = 0
        frustrations = 0
        
        frustration_phrases = [
            "no,", "don't", "stop", "that's not", "i said", "actually",
            "wrong", "incorrect", "you misunderstood"
        ]
        
        for conv in conversations[:100]:
            for turn in conv.get("turns", []):
                if turn.get("role") == "user":
                    content = turn.get("content", "").lower()
                    if any(phrase in content for phrase in frustration_phrases):
                        if len(content) < 100:  # Short corrections
                            corrections += 1
                        else:
                            frustrations += 1
        
        if corrections > 5:
            self._user_preferences["correction_tolerance"] = "low - execute correctly first time"
        if frustrations > 3:
            self._user_preferences["directness"] = "high - avoid hedging or asking"
    
    def _format_examples(self, examples: List[Dict]) -> str:
        """Format examples for inclusion in context."""
        parts = []
        for i, ex in enumerate(examples, 1):
            prompt = ex["prompt"][:200] + "..." if len(ex["prompt"]) > 200 else ex["prompt"]
            response = ex["response"][:300] + "..." if len(ex["response"]) > 300 else ex["response"]
            parts.append(f"Example {i}:")
            parts.append(f"  User: {prompt}")
            parts.append(f"  Assistant: {response}")
        return "\n".join(parts)
    
    def _default_style(self) -> str:
        """Return default style signature."""
        return """Preferred Response Style:
- Be direct and execute immediately
- Use numbered lists for multi-step processes
- Include code blocks for technical content
- Never ask clarifying questions unless absolutely necessary
- Provide complete implementations without omissions"""
    
    def get_domain_context(self, domain: str) -> str:
        """Get context specific to a domain."""
        examples = self._domain_examples.get(domain, [])
        if not examples:
            return ""
        
        return f"## {domain.upper()} EXAMPLES\n" + self._format_examples(examples[:self.max_examples])

