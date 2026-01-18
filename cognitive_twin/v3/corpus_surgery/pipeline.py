"""
Corpus Surgery Pipeline for CognitiveTwin V3.

Orchestrates the full corpus surgery process:
1. Load conversations from Supabase or JSONL
2. Classify each assistant turn
3. Rewrite unjustified clarifications
4. Quarantine friction segments
5. Export SFT data, DPO pairs, and eval cases
"""

import json
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from .types import (
    ClarificationType,
    ClassificationResult,
    ProcessedTurn,
    ProcessedConversation,
    DPOPair,
    EvalCase,
    FormatConstraints,
)
from .classifier import (
    classify_assistant_turn,
    extract_format_constraints,
    compute_directive_completeness,
)
from .rewriter import (
    rewrite_assistant_turn,
    should_rewrite,
    validate_rewrite,
)
from .quarantine import (
    detect_frustration,
    scan_conversation_for_friction,
    create_dpo_pair_from_quarantine,
    create_eval_case_from_quarantine,
    generate_ideal_response,
    export_quarantine_stats,
)
from ..api.openai_client import V3OpenAIClient, ClientConfig


@dataclass
class PipelineConfig:
    """Configuration for the corpus surgery pipeline."""
    
    # Input/Output
    input_path: Optional[str] = None
    output_dir: str = "data/corpus_surgery_output"
    
    # Processing
    enable_rewriting: bool = True
    enable_quarantine: bool = True
    generate_ideal_responses: bool = True
    
    # Concurrency
    max_concurrent_rewrites: int = 5
    max_concurrent_quarantine: int = 5
    
    # Filtering
    min_conversation_turns: int = 4
    skip_neutral_turns: bool = False
    
    # Output
    export_sft: bool = True
    export_dpo: bool = True
    export_eval: bool = True
    export_stats: bool = True
    
    # API
    openai_config: Optional[ClientConfig] = None


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""
    
    total_conversations: int = 0
    total_turns: int = 0
    
    # Classification
    unjustified_count: int = 0
    justified_count: int = 0
    neutral_count: int = 0
    
    # Actions
    kept_count: int = 0
    rewritten_count: int = 0
    rewrite_failures: int = 0
    quarantined_count: int = 0
    
    # Outputs
    sft_examples: int = 0
    dpo_pairs: int = 0
    eval_cases: int = 0
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_conversations": self.total_conversations,
            "total_turns": self.total_turns,
            "classification": {
                "unjustified": self.unjustified_count,
                "justified": self.justified_count,
                "neutral": self.neutral_count,
            },
            "actions": {
                "kept": self.kept_count,
                "rewritten": self.rewritten_count,
                "rewrite_failures": self.rewrite_failures,
                "quarantined": self.quarantined_count,
            },
            "outputs": {
                "sft_examples": self.sft_examples,
                "dpo_pairs": self.dpo_pairs,
                "eval_cases": self.eval_cases,
            },
            "timing": {
                "start": self.start_time,
                "end": self.end_time,
                "duration_seconds": self.duration_seconds,
            },
        }


class CorpusSurgeryPipeline:
    """Main pipeline for corpus surgery."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()
        
        # Initialize client
        openai_config = self.config.openai_config or ClientConfig()
        self.client = V3OpenAIClient(openai_config)
        
        # Output storage
        self.sft_data: List[dict] = []
        self.dpo_pairs: List[DPOPair] = []
        self.eval_cases: List[EvalCase] = []
    
    async def process_turn(
        self,
        assistant_message: str,
        user_message: str,
        conversation_history: List[dict],
        phase_id: int = 2,
        next_user_message: Optional[str] = None,
    ) -> ProcessedTurn:
        """
        Process a single assistant turn.
        
        Args:
            assistant_message: The assistant's response
            user_message: The preceding user message
            conversation_history: Prior conversation
            phase_id: Conversation phase
            next_user_message: Following user message (for friction detection)
        
        Returns:
            ProcessedTurn with classification and action taken
        """
        # Extract format constraints
        format_constraints = extract_format_constraints(user_message)
        
        # Compute directive completeness
        directive_completeness = compute_directive_completeness(user_message)
        
        # Classify
        classification = classify_assistant_turn(
            assistant_message=assistant_message,
            user_message=user_message,
            phase_id=phase_id,
            format_constraints=format_constraints.to_dict(),
            directive_completeness=directive_completeness,
        )
        
        # Update stats
        if classification.classification == ClarificationType.UNJUSTIFIED:
            self.stats.unjustified_count += 1
        elif classification.classification == ClarificationType.JUSTIFIED:
            self.stats.justified_count += 1
        else:
            self.stats.neutral_count += 1
        
        # Determine action
        action_taken = "kept"
        processed_content = assistant_message
        rewrite_valid = True
        rewrite_errors = []
        quarantine_marker = None
        dpo_pair = None
        eval_case = None
        
        # Check for friction (if next user message available)
        is_friction = False
        if next_user_message:
            is_friction, _ = detect_frustration(next_user_message)
        
        # Rewrite if unjustified and no friction
        if (self.config.enable_rewriting and 
            should_rewrite(classification) and 
            not is_friction):
            
            try:
                processed_content = await rewrite_assistant_turn(
                    assistant_message=assistant_message,
                    user_message=user_message,
                    conversation_history=conversation_history,
                    format_constraints=format_constraints,
                    client=self.client,
                )
                
                validation = validate_rewrite(
                    processed_content, user_message, format_constraints
                )
                
                rewrite_valid = validation.is_valid
                rewrite_errors = validation.errors
                action_taken = "rewritten"
                self.stats.rewritten_count += 1
                
            except Exception as e:
                rewrite_errors = [str(e)]
                rewrite_valid = False
                self.stats.rewrite_failures += 1
                action_taken = "kept"  # Keep original if rewrite fails
        
        # Keep if not rewritten
        if action_taken == "kept":
            self.stats.kept_count += 1
        
        return ProcessedTurn(
            original_content=assistant_message,
            processed_content=processed_content,
            classification=classification,
            action_taken=action_taken,
            rewrite_valid=rewrite_valid,
            rewrite_errors=rewrite_errors,
            quarantine_marker=quarantine_marker,
            dpo_pair=dpo_pair,
            eval_case=eval_case,
        )
    
    async def process_conversation(
        self,
        conversation: List[dict],
        conversation_id: str,
    ) -> ProcessedConversation:
        """
        Process a full conversation.
        
        Args:
            conversation: List of turn dicts with 'role' and 'content'
            conversation_id: Unique identifier
        
        Returns:
            ProcessedConversation with all processed turns
        """
        processed_turns = []
        sft_turns = []
        
        # First pass: process turns
        for i, turn in enumerate(conversation):
            if turn.get("role") != "assistant":
                continue
            
            # Find preceding user message
            user_message = ""
            history = []
            for j in range(i - 1, -1, -1):
                if conversation[j].get("role") == "user" and not user_message:
                    user_message = conversation[j].get("content", "")
                history.insert(0, conversation[j])
            
            if not user_message:
                continue
            
            # Find next user message (for friction detection)
            next_user_message = None
            for j in range(i + 1, len(conversation)):
                if conversation[j].get("role") == "user":
                    next_user_message = conversation[j].get("content", "")
                    break
            
            # Process turn
            processed = await self.process_turn(
                assistant_message=turn.get("content", ""),
                user_message=user_message,
                conversation_history=history[:6],
                next_user_message=next_user_message,
            )
            
            processed_turns.append(processed)
            self.stats.total_turns += 1
            
            # Add to SFT data if kept or successfully rewritten
            if processed.action_taken in ["kept", "rewritten"]:
                if processed.action_taken == "rewritten" and not processed.rewrite_valid:
                    continue  # Skip invalid rewrites
                
                sft_turn = {
                    "instruction": user_message,
                    "output": processed.processed_content,
                    "history": [
                        {"role": t.get("role"), "content": t.get("content")}
                        for t in history[-4:]
                    ],
                    "classification": processed.classification.classification.value,
                    "was_rewritten": processed.action_taken == "rewritten",
                }
                sft_turns.append(sft_turn)
                self.sft_data.append(sft_turn)
        
        # Second pass: quarantine friction segments
        dpo_pairs = []
        eval_cases = []
        
        if self.config.enable_quarantine:
            from .quarantine import process_conversation_for_quarantine
            
            quarantine_result = await process_conversation_for_quarantine(
                conversation=conversation,
                conversation_id=conversation_id,
                client=self.client,
                generate_ideal=self.config.generate_ideal_responses,
            )
            
            dpo_pairs = quarantine_result.get("dpo_pairs", [])
            eval_cases = quarantine_result.get("eval_cases", [])
            
            self.dpo_pairs.extend(dpo_pairs)
            self.eval_cases.extend(eval_cases)
            self.stats.quarantined_count += quarantine_result.get("friction_count", 0)
        
        # Count outputs
        kept_count = sum(1 for t in processed_turns if t.action_taken == "kept")
        rewritten_count = sum(1 for t in processed_turns if t.action_taken == "rewritten")
        
        return ProcessedConversation(
            conversation_id=conversation_id,
            original_turns=len(conversation),
            processed_turns=processed_turns,
            kept_count=kept_count,
            rewritten_count=rewritten_count,
            quarantined_count=len(dpo_pairs),
            sft_turns=sft_turns,
            dpo_pairs=dpo_pairs,
            eval_cases=eval_cases,
        )
    
    async def process_batch(
        self,
        conversations: List[Tuple[List[dict], str]],
    ) -> List[ProcessedConversation]:
        """
        Process multiple conversations.
        
        Args:
            conversations: List of (conversation, id) tuples
        
        Returns:
            List of ProcessedConversation objects
        """
        results = []
        
        for conversation, conv_id in conversations:
            if len(conversation) < self.config.min_conversation_turns:
                continue
            
            result = await self.process_conversation(conversation, conv_id)
            results.append(result)
            self.stats.total_conversations += 1
        
        return results
    
    def load_conversations_from_jsonl(
        self,
        path: str,
    ) -> List[Tuple[List[dict], str]]:
        """Load conversations from JSONL file."""
        conversations = []
        
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    conv = data.get("conversation", data.get("messages", []))
                    conv_id = data.get("id", f"conv_{i}")
                    conversations.append((conv, conv_id))
                except json.JSONDecodeError:
                    continue
        
        return conversations
    
    def export_results(self, output_dir: Optional[str] = None):
        """Export all results to files."""
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export SFT data
        if self.config.export_sft and self.sft_data:
            sft_path = output_path / f"train_sft_{timestamp}.jsonl"
            with open(sft_path, 'w') as f:
                for example in self.sft_data:
                    f.write(json.dumps(example) + '\n')
            self.stats.sft_examples = len(self.sft_data)
            print(f"Exported {len(self.sft_data)} SFT examples to {sft_path}")
        
        # Export DPO pairs
        if self.config.export_dpo and self.dpo_pairs:
            dpo_path = output_path / f"train_dpo_{timestamp}.jsonl"
            with open(dpo_path, 'w') as f:
                for pair in self.dpo_pairs:
                    f.write(json.dumps(pair.to_training_format()) + '\n')
            self.stats.dpo_pairs = len(self.dpo_pairs)
            print(f"Exported {len(self.dpo_pairs)} DPO pairs to {dpo_path}")
        
        # Export eval cases
        if self.config.export_eval and self.eval_cases:
            eval_path = output_path / f"eval_regression_{timestamp}.jsonl"
            with open(eval_path, 'w') as f:
                for case in self.eval_cases:
                    f.write(json.dumps(case.to_dict()) + '\n')
            self.stats.eval_cases = len(self.eval_cases)
            print(f"Exported {len(self.eval_cases)} eval cases to {eval_path}")
        
        # Export stats
        if self.config.export_stats:
            stats_path = output_path / f"stats_{timestamp}.json"
            with open(stats_path, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            print(f"Exported stats to {stats_path}")
    
    async def run(
        self,
        conversations: Optional[List[Tuple[List[dict], str]]] = None,
    ) -> PipelineStats:
        """
        Run the full pipeline.
        
        Args:
            conversations: Conversations to process (or load from config.input_path)
        
        Returns:
            PipelineStats with execution results
        """
        self.stats.start_time = datetime.now().isoformat()
        start = datetime.now()
        
        # Load conversations if not provided
        if conversations is None:
            if self.config.input_path:
                conversations = self.load_conversations_from_jsonl(self.config.input_path)
            else:
                raise ValueError("No conversations provided and no input_path configured")
        
        print(f"Processing {len(conversations)} conversations...")
        
        # Process
        await self.process_batch(conversations)
        
        # Export
        self.export_results()
        
        # Finalize stats
        self.stats.end_time = datetime.now().isoformat()
        self.stats.duration_seconds = (datetime.now() - start).total_seconds()
        
        print(f"\nPipeline complete in {self.stats.duration_seconds:.2f}s")
        print(f"  Conversations: {self.stats.total_conversations}")
        print(f"  Turns processed: {self.stats.total_turns}")
        print(f"  Unjustified: {self.stats.unjustified_count}")
        print(f"  Rewritten: {self.stats.rewritten_count}")
        print(f"  Quarantined: {self.stats.quarantined_count}")
        
        return self.stats


def run_pipeline_sync(
    conversations: Optional[List[Tuple[List[dict], str]]] = None,
    config: Optional[PipelineConfig] = None,
) -> PipelineStats:
    """Synchronous wrapper for pipeline execution."""
    pipeline = CorpusSurgeryPipeline(config)
    return asyncio.run(pipeline.run(conversations))


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Corpus Surgery Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", default="data/corpus_surgery_output", help="Output directory")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable rewriting")
    parser.add_argument("--no-quarantine", action="store_true", help="Disable quarantine")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output,
        enable_rewriting=not args.no_rewrite,
        enable_quarantine=not args.no_quarantine,
    )
    
    stats = run_pipeline_sync(config=config)
    print("\nFinal stats:")
    print(json.dumps(stats.to_dict(), indent=2))

