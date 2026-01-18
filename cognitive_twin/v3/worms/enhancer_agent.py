"""
Enhancer Agent for CognitiveTwin V3.

Main orchestrator that:
- Canonicalizes messy outputs
- Completes unfinished code/prose
- Extracts regression test cases from annoyances
- Creates SFT records and DPO pairs

Usage:
    python -m rag_plusplus.ml.cognitivetwin_v3.worms.enhancer_agent \
        --output data/enhancer_output \
        --max-conversations 100
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .enhancer_types import (
    EnhancerConfig,
    EvalCase,
    AnnoyanceRecord,
    EnhancedRecord,
    EnhancerStats,
)
from .canonicalizer import Canonicalizer
from .completer import Completer
from .annoyance_detector import AnnoyanceDetector


logger = logging.getLogger(__name__)


class EnhancerAgent:
    """Enhancer Agent for canonicalization, completion, and eval extraction."""
    
    def __init__(
        self,
        supabase_client=None,
        config: Optional[EnhancerConfig] = None,
        openai_client=None,
    ):
        self.supabase = supabase_client
        self.config = config or EnhancerConfig()
        
        # Initialize OpenAI client
        if openai_client is None:
            try:
                from ..api.openai_client import V3OpenAIClient
                self.openai = V3OpenAIClient()
            except ImportError:
                logger.warning("V3OpenAIClient not available, using stub")
                self.openai = None
        else:
            self.openai = openai_client
        
        # Initialize components
        self.canonicalizer = Canonicalizer(
            max_apologies=self.config.max_apologies_to_keep,
            keep_sensitive_disclaimers=self.config.keep_sensitive_disclaimers,
            sensitive_topics=self.config.sensitive_topics,
        )
        self.completer = Completer(self.openai)
        self.detector = AnnoyanceDetector(self.openai, self.config)
        
        # Stats tracking
        self.stats = EnhancerStats()
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    async def enhance_conversation(
        self,
        conversation: dict,
    ) -> list[EnhancedRecord]:
        """Enhance all assistant turns in a conversation."""
        enhanced_records = []
        turns = conversation.get("turns", [])
        
        for i, turn in enumerate(turns):
            if turn.get("role") != "assistant":
                continue
            
            original = turn.get("content", "")
            enhanced = original
            changes = []
            
            # Get context for enhancement
            context = self._get_context_string(turns, i)
            
            # Canonicalize
            if self.config.remove_provider_isms:
                enhanced, canon_changes = self.canonicalizer.canonicalize(enhanced, context)
                changes.extend(canon_changes)
                
                # Track stats
                if "provider-ism" in str(canon_changes).lower():
                    self.stats.provider_isms_removed += 1
                if "filler opening" in str(canon_changes).lower():
                    self.stats.filler_openings_removed += 1
                if "permission closer" in str(canon_changes).lower():
                    self.stats.permission_closers_removed += 1
                if "apolog" in str(canon_changes).lower():
                    self.stats.apologies_reduced += 1
            
            # Complete unfinished content
            if self.config.complete_code or self.config.complete_prose:
                if self.completer.has_incomplete_content(enhanced):
                    enhanced, code_count, prose_count = await self.completer.complete_all(
                        enhanced, context
                    )
                    
                    if code_count:
                        changes.append(f"Completed {code_count} code block(s)")
                        self.stats.code_blocks_completed += code_count
                    if prose_count:
                        changes.append(f"Completed {prose_count} prose section(s)")
                        self.stats.prose_sections_completed += prose_count
            
            # Create record if enhanced
            if enhanced != original:
                # Determine enhancement type
                has_canonicalization = any(
                    "removed" in c.lower() or "reduced" in c.lower()
                    for c in changes
                )
                has_completion = any(
                    "completed" in c.lower()
                    for c in changes
                )
                
                if has_canonicalization and has_completion:
                    enhancement_type = "both"
                elif has_completion:
                    enhancement_type = "completion"
                else:
                    enhancement_type = "canonicalization"
                
                # Get context messages
                context_messages = [
                    {"role": t.get("role"), "content": t.get("content")}
                    for t in turns[:i]
                ]
                
                record = EnhancedRecord(
                    original=original,
                    enhanced=enhanced,
                    enhancement_type=enhancement_type,
                    conversation_id=conversation.get("id", ""),
                    turn_index=i,
                    changes_made=changes,
                    context=context_messages,
                )
                
                enhanced_records.append(record)
                
                # Update the turn for downstream processing
                turn["content"] = enhanced
        
        self.stats.turns_processed += len([t for t in turns if t.get("role") == "assistant"])
        
        return enhanced_records
    
    async def process_batch(
        self,
        conversations: list[dict],
    ) -> tuple[list[EnhancedRecord], list[EvalCase], list[dict]]:
        """Process a batch of conversations.
        
        Returns:
            Tuple of (enhanced_records, eval_cases, dpo_pairs)
        """
        all_enhanced = []
        all_eval_cases = []
        all_dpo_pairs = []
        
        # Phase 1: Enhance all conversations
        for conv in conversations:
            enhanced = await self.enhance_conversation(conv)
            all_enhanced.extend(enhanced)
            self.stats.conversations_processed += 1
        
        # Phase 2: Extract annoyances
        if self.config.extract_eval_cases:
            annoyances = self.detector.find_all_annoyances(conversations)
            
            # Track stats
            for ann in annoyances:
                if ann.type == "permission_seeking":
                    self.stats.permission_annoyances += 1
                elif ann.type == "omission":
                    self.stats.omission_annoyances += 1
                elif ann.type == "format_drift":
                    self.stats.format_drift_annoyances += 1
            
            # Create eval cases
            for annoyance in annoyances:
                eval_case = self.detector.create_eval_case_from_annoyance(annoyance)
                
                # Generate reference answer
                if self.openai:
                    eval_case.reference_answer = await self.detector.generate_reference_answer(
                        eval_case
                    )
                
                all_eval_cases.append(eval_case)
                
                # Create DPO pair
                if eval_case.reference_answer:
                    dpo_pair = self.create_dpo_pair(annoyance, eval_case.reference_answer)
                    all_dpo_pairs.append(dpo_pair)
        
        # Update stats
        self.stats.enhanced_records = len(all_enhanced)
        self.stats.eval_cases = len(all_eval_cases)
        self.stats.dpo_pairs = len(all_dpo_pairs)
        
        return all_enhanced, all_eval_cases, all_dpo_pairs
    
    def _get_context_string(
        self,
        turns: list[dict],
        current_index: int,
    ) -> str:
        """Get context string from preceding turns."""
        context_turns = turns[max(0, current_index - 4):current_index]
        return "\n".join([
            f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')[:300]}"
            for t in context_turns
        ])
    
    # =========================================================================
    # RECORD CREATION
    # =========================================================================
    
    def create_enhanced_sft_record(
        self,
        record: EnhancedRecord,
    ) -> dict:
        """Create SFT record from enhanced content."""
        return {
            "schema_version": "ctv3.1",
            "record_id": str(uuid4()),
            "record_type": "sft_turn",
            "source": {
                "origin": "enhancer_agent",
                "provider": "gpt-5.2",
                "source_id": record.conversation_id,
                "created_at_utc": datetime.utcnow().isoformat(),
            },
            "context": {
                "domain": "mixed",
                "policy": {
                    "question_policy": "no_questions",
                    "directive_completeness": 0.8,
                },
            },
            "input": {
                "messages": record.context,
                "attachments": [],
            },
            "target": {
                "assistant_content": record.enhanced,
                "structured": {},
            },
            "tags": {
                "task_type": "respond",
                "enhancement_type": record.enhancement_type,
                "changes_made": record.changes_made,
            },
            "quality": {
                "gold": True,
                "weight": 1.0,
                "review_status": "auto",
            },
        }
    
    def export_eval_case(self, case: EvalCase) -> dict:
        """Export eval case to CTv3.1 format."""
        return {
            "schema_version": "ctv3.1",
            "record_id": case.case_id,
            "record_type": "eval_case",
            "source": {
                "origin": "enhancer_agent",
                "source_id": case.source_conversation,
            },
            "input": {
                "messages": case.context + [{"role": "user", "content": case.prompt}],
            },
            "context": {
                "policy": {
                    "question_policy": "no_questions",
                },
                "format_constraints": case.format_constraints,
            },
            "checks": {
                "expected_behaviors": case.expected_behaviors,
                "disallowed_behaviors": case.disallowed_behaviors,
                "disallowed_phrases": case.disallowed_phrases,
                "must_not_end_with_question": case.must_not_end_with_question,
                "must_follow_format": case.must_follow_format,
            },
            "reference": {
                "answer": case.reference_answer,
            },
            "quality": {
                "gold": True,
                "weight": 0.0,  # Not trained on
                "review_status": "auto",
            },
        }
    
    def create_dpo_pair(
        self,
        annoyance: AnnoyanceRecord,
        reference_answer: str,
    ) -> dict:
        """Create DPO pair from annoyance and reference answer."""
        return {
            "schema_version": "ctv3.1",
            "record_id": str(uuid4()),
            "record_type": "dpo_pair",
            "source": {
                "origin": "enhancer_agent",
                "provider": "gpt-5.2",
                "source_id": annoyance.conversation_id,
                "created_at_utc": datetime.utcnow().isoformat(),
            },
            "context": {
                "domain": "mixed",
                "policy": {
                    "question_policy": "no_questions",
                },
            },
            "input": {
                "messages": [{"role": "user", "content": annoyance.user_message}],
            },
            "candidates": {
                "preferred": {"assistant_content": reference_answer},
                "dispreferred": {"assistant_content": annoyance.assistant_message},
            },
            "tags": {
                "task_type": "respond",
                "dpo_reason": annoyance.type,
            },
            "quality": {
                "gold": True,
                "weight": 1.0,
                "review_status": "auto",
            },
        }
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_enhanced_records(
        self,
        records: list[EnhancedRecord],
        output_path: Path,
    ):
        """Export enhanced records to JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for record in records:
                sft_record = self.create_enhanced_sft_record(record)
                f.write(json.dumps(sft_record) + '\n')
        
        logger.info(f"Exported {len(records)} enhanced records to {output_path}")
    
    def export_eval_cases(
        self,
        cases: list[EvalCase],
        output_path: Path,
    ):
        """Export eval cases to JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for case in cases:
                exported = self.export_eval_case(case)
                f.write(json.dumps(exported) + '\n')
        
        logger.info(f"Exported {len(cases)} eval cases to {output_path}")
    
    def export_dpo_pairs(
        self,
        pairs: list[dict],
        output_path: Path,
    ):
        """Export DPO pairs to JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
        
        logger.info(f"Exported {len(pairs)} DPO pairs to {output_path}")


class EnhancerAgentPipeline:
    """Complete Enhancer Agent pipeline."""
    
    def __init__(
        self,
        supabase_client=None,
        config: Optional[EnhancerConfig] = None,
    ):
        self.supabase = supabase_client
        self.config = config or EnhancerConfig()
    
    async def run(
        self,
        conversations: Optional[list[dict]] = None,
        output_dir: Optional[Path] = None,
        max_conversations: int = 100,
    ) -> EnhancerStats:
        """Run the complete pipeline."""
        # Initialize agent
        agent = EnhancerAgent(self.supabase, self.config)
        agent.stats.start_time = datetime.utcnow().isoformat()
        
        # Load conversations if not provided
        if conversations is None:
            conversations = await self._load_conversations(max_conversations)
        else:
            conversations = conversations[:max_conversations]
        
        if not conversations:
            logger.warning("No conversations to process")
            agent.stats.end_time = datetime.utcnow().isoformat()
            return agent.stats
        
        logger.info(f"Processing {len(conversations)} conversations")
        
        # Process
        enhanced_records, eval_cases, dpo_pairs = await agent.process_batch(conversations)
        
        # Export
        if output_dir:
            output_dir = Path(output_dir)
            agent.export_enhanced_records(enhanced_records, output_dir / "enhanced_sft.jsonl")
            agent.export_eval_cases(eval_cases, output_dir / "eval_regression.jsonl")
            agent.export_dpo_pairs(dpo_pairs, output_dir / "enhancer_dpo.jsonl")
            
            # Write stats
            with open(output_dir / "enhancer_stats.json", 'w') as f:
                json.dump(agent.stats.to_dict(), f, indent=2)
        
        agent.stats.end_time = datetime.utcnow().isoformat()
        
        if agent.stats.start_time and agent.stats.end_time:
            start = datetime.fromisoformat(agent.stats.start_time)
            end = datetime.fromisoformat(agent.stats.end_time)
            agent.stats.duration_seconds = (end - start).total_seconds()
        
        logger.info(f"Pipeline complete: {agent.stats.to_dict()}")
        
        return agent.stats
    
    async def _load_conversations(self, limit: int) -> list[dict]:
        """Load conversations from Supabase."""
        if not self.supabase:
            logger.warning("No Supabase client, returning empty conversations")
            return []
        
        try:
            # Get unique conversation IDs
            response = self.supabase.table("memory_turns") \
                .select("conversation_id") \
                .limit(limit * 10) \
                .execute()
            
            conv_ids = list(set(
                turn.get("conversation_id")
                for turn in response.data
                if turn.get("conversation_id")
            ))[:limit]
            
            # Load full conversations
            conversations = []
            for conv_id in conv_ids:
                turns_response = self.supabase.table("memory_turns") \
                    .select("*") \
                    .eq("conversation_id", conv_id) \
                    .order("created_at", desc=False) \
                    .execute()
                
                turns = [
                    {
                        "role": t.get("role", "user"),
                        "content": t.get("content", ""),
                    }
                    for t in turns_response.data
                ]
                
                if turns:
                    conversations.append({
                        "id": conv_id,
                        "turns": turns,
                    })
            
            return conversations
        
        except Exception as e:
            logger.warning(f"Failed to load conversations: {e}")
            return []


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhancer Agent for CognitiveTwin V3"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/enhancer_output"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-conversations", "-n",
        type=int,
        default=100,
        help="Maximum conversations to process",
    )
    parser.add_argument(
        "--skip-completion",
        action="store_true",
        help="Skip code/prose completion",
    )
    parser.add_argument(
        "--skip-eval-extraction",
        action="store_true",
        help="Skip eval case extraction",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Configure
    config = EnhancerConfig(
        complete_code=not args.skip_completion,
        complete_prose=not args.skip_completion,
        extract_eval_cases=not args.skip_eval_extraction,
    )
    
    # Initialize Supabase client
    supabase = None
    try:
        import os
        from supabase import create_client
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if url and key:
            supabase = create_client(url, key)
            logger.info("Supabase client initialized")
    except ImportError:
        logger.warning("Supabase not available")
    
    # Run pipeline
    pipeline = EnhancerAgentPipeline(supabase, config)
    
    stats = await pipeline.run(
        output_dir=args.output,
        max_conversations=args.max_conversations,
    )
    
    print("\n" + "=" * 60)
    print("Enhancer Agent Complete")
    print("=" * 60)
    print(f"Conversations processed: {stats.conversations_processed}")
    print(f"Turns processed: {stats.turns_processed}")
    print()
    print("Canonicalization:")
    print(f"  - Provider-isms removed: {stats.provider_isms_removed}")
    print(f"  - Filler openings removed: {stats.filler_openings_removed}")
    print(f"  - Permission closers removed: {stats.permission_closers_removed}")
    print(f"  - Apologies reduced: {stats.apologies_reduced}")
    print()
    print("Completion:")
    print(f"  - Code blocks completed: {stats.code_blocks_completed}")
    print(f"  - Prose sections completed: {stats.prose_sections_completed}")
    print()
    print("Annoyances found:")
    print(f"  - Permission-seeking: {stats.permission_annoyances}")
    print(f"  - Omissions: {stats.omission_annoyances}")
    print(f"  - Format drift: {stats.format_drift_annoyances}")
    print()
    print(f"Enhanced records: {stats.enhanced_records}")
    print(f"Eval cases: {stats.eval_cases}")
    print(f"DPO pairs: {stats.dpo_pairs}")
    print(f"Duration: {stats.duration_seconds:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

