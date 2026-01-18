"""
Conversation Worm for CognitiveTwin V3.

Traverses conversation DAGs to generate topology-consistent synthetic dialogues:
- Paraphrase variants for robustness
- Ideal responses to fix friction points
- Trajectory-preserving extensions
- Phase-aware training data

Usage:
    python -m rag_plusplus.ml.cognitivetwin_v3.worms.conversation_worm \
        --output data/convo_worm_output \
        --max-conversations 100
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .branch_types import (
    ConversationWormConfig,
    SyntheticBranch,
    DLMCoordinate,
    PathNode,
    ConversationPath,
    BranchResult,
    ConversationWormStats,
)
from .branch_generator import BranchGenerator
from .policy_enforcer import PolicyEnforcer


logger = logging.getLogger(__name__)


class ConversationWorm:
    """Conversation DAG traversal agent for training data generation."""
    
    def __init__(
        self,
        supabase_client=None,
        config: Optional[ConversationWormConfig] = None,
        openai_client=None,
        v2_generator=None,
        use_v2: bool = True,
    ):
        self.supabase = supabase_client
        self.config = config or ConversationWormConfig()
        self.use_v2 = use_v2
        
        # Initialize V2 generator (preferred for style-aligned outputs)
        self.v2_generator = v2_generator
        if use_v2 and v2_generator is None:
            try:
                from ..generators.v2_generator import V2Generator
                self.v2_generator = V2Generator()
                logger.info("Using V2Generator for conversation branching")
            except ImportError:
                logger.warning("V2Generator not available, falling back to OpenAI")
                self.use_v2 = False
        
        # Initialize OpenAI client (fallback)
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
        self.generator = BranchGenerator(self.openai, self.config, self.v2_generator)
        self.enforcer = PolicyEnforcer(self.config)
        
        # State
        self.processed_conversations: set[str] = set()
        self.generated_branches: list[SyntheticBranch] = []
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    async def process_conversation(
        self,
        conversation_id: str,
    ) -> BranchResult:
        """Process a single conversation."""
        result = BranchResult(conversation_id=conversation_id)
        
        try:
            # Load conversation
            messages = await self._load_conversation(conversation_id)
            
            if not messages:
                result.success = False
                result.error = "No messages found"
                return result
            
            # Build paths from messages
            paths = self._build_paths(conversation_id, messages)
            
            # Identify friction paths
            friction_paths = self._identify_friction_paths(paths)
            result.friction_points_found = len(friction_paths)
            
            # Generate ideal branches for friction
            for path in friction_paths[:self.config.max_branches_per_friction]:
                ideal_branch = await self.generator.create_ideal_branch(path)
                if ideal_branch:
                    result.ideal_branches.append(ideal_branch)
            
            # Generate paraphrases for all paths
            for path in paths:
                paraphrase_branches = await self.generator.create_paraphrase_branches(path)
                result.paraphrase_branches.extend(paraphrase_branches)
            
            # Generate extensions for high-quality paths
            extension_branches = await self.generator.create_extension_branches(
                paths, self.config.min_quality_threshold
            )
            result.extension_branches.extend(extension_branches)
            
            # Update totals
            result.total_branches = len(result.all_branches)
            
            # Track
            self.generated_branches.extend(result.all_branches)
            self.processed_conversations.add(conversation_id)
            
        except Exception as e:
            logger.exception(f"Error processing conversation {conversation_id}")
            result.success = False
            result.error = str(e)
        
        return result
    
    async def process_batch(
        self,
        conversation_ids: list[str],
    ) -> list[BranchResult]:
        """Process multiple conversations."""
        results = []
        
        # Process in chunks to respect concurrency limit
        chunk_size = self.config.max_concurrent_conversations
        
        for i in range(0, len(conversation_ids), chunk_size):
            chunk = conversation_ids[i:i + chunk_size]
            
            tasks = [
                self.process_conversation(conv_id)
                for conv_id in chunk
            ]
            
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for conv_id, result in zip(chunk, chunk_results):
                if isinstance(result, Exception):
                    results.append(BranchResult(
                        conversation_id=conv_id,
                        success=False,
                        error=str(result),
                    ))
                else:
                    results.append(result)
        
        return results
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    async def _load_conversation(
        self,
        conversation_id: str,
    ) -> list[dict]:
        """Load conversation messages from Supabase."""
        if not self.supabase:
            logger.warning("No Supabase client, returning empty messages")
            return []
        
        try:
            response = self.supabase.table("memory_turns") \
                .select("*") \
                .eq("conversation_id", conversation_id) \
                .order("created_at", desc=False) \
                .execute()
            
            messages = []
            for turn in response.data:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                    "turn_id": turn.get("id", ""),
                    "created_at": turn.get("created_at", ""),
                })
            
            return messages
        
        except Exception as e:
            logger.warning(f"Failed to load conversation {conversation_id}: {e}")
            return []
    
    async def get_all_conversation_ids(
        self,
        limit: int = 1000,
    ) -> list[str]:
        """Get all conversation IDs from Supabase."""
        if not self.supabase:
            return []
        
        try:
            response = self.supabase.table("memory_turns") \
                .select("conversation_id") \
                .limit(limit * 10) \
                .execute()
            
            # Get unique conversation IDs
            conv_ids = list(set(
                turn.get("conversation_id")
                for turn in response.data
                if turn.get("conversation_id")
            ))
            
            return conv_ids[:limit]
        
        except Exception as e:
            logger.warning(f"Failed to get conversation IDs: {e}")
            return []
    
    # =========================================================================
    # PATH BUILDING
    # =========================================================================
    
    def _build_paths(
        self,
        conversation_id: str,
        messages: list[dict],
    ) -> list[ConversationPath]:
        """Build ConversationPath objects from messages."""
        if not messages:
            return []
        
        # Build nodes
        nodes = []
        for i, msg in enumerate(messages):
            node = PathNode(
                turn_id=msg.get("turn_id", f"turn_{i}"),
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                depth=i,
                sibling_order=0,
                temporal=i / len(messages) if messages else 0,
                phase_id=self.enforcer.estimate_phase(messages, i),
            )
            nodes.append(node)
        
        # Compute quality score
        quality = self._compute_path_quality(nodes)
        
        # Check for friction
        has_friction = any(
            self.enforcer.is_repair_turn(
                node.content,
                nodes[i - 1].content if i > 0 else ""
            )
            for i, node in enumerate(nodes)
            if node.role == "user"
        )
        
        # Create single path (for linear conversations)
        # For branched conversations, would create multiple paths
        path = ConversationPath(
            conversation_id=conversation_id,
            nodes=nodes,
            quality_score=quality,
            has_friction=has_friction,
        )
        
        return [path]
    
    def _compute_path_quality(self, nodes: list[PathNode]) -> float:
        """Compute quality score for a path."""
        if not nodes:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length bonus
        if len(nodes) >= 4:
            score += 0.1
        if len(nodes) >= 8:
            score += 0.1
        
        # Check for friction (reduces quality)
        friction_count = 0
        for i, node in enumerate(nodes):
            if node.role == "user" and i > 0:
                if self.enforcer.is_repair_turn(
                    node.content,
                    nodes[i - 1].content
                ):
                    friction_count += 1
        
        score -= friction_count * 0.15
        
        return max(0.0, min(1.0, score))
    
    # =========================================================================
    # FRICTION DETECTION
    # =========================================================================
    
    def _identify_friction_paths(
        self,
        paths: list[ConversationPath],
    ) -> list[ConversationPath]:
        """Identify paths that contain friction points."""
        friction_paths = []
        
        for path in paths:
            has_friction = False
            
            for i, node in enumerate(path.nodes):
                # Check for user frustration
                if node.role == "user":
                    try:
                        from ..corpus_surgery.quarantine import detect_frustration
                        if detect_frustration(node.content):
                            has_friction = True
                            break
                    except ImportError:
                        # Fallback
                        if self.enforcer.is_repair_turn(node.content, ""):
                            has_friction = True
                            break
                
                # Check for unjustified clarification
                if node.role == "assistant":
                    user_content = ""
                    for j in range(i - 1, -1, -1):
                        if path.nodes[j].role == "user":
                            user_content = path.nodes[j].content
                            break
                    
                    completeness = self.enforcer.compute_directive_completeness(
                        user_content
                    )
                    
                    if completeness >= 0.5:
                        stall_score = self.enforcer._compute_stall_score(node.content)
                        if stall_score >= 3:
                            has_friction = True
                            break
            
            if has_friction:
                path.has_friction = True
                friction_paths.append(path)
        
        return friction_paths
    
    # =========================================================================
    # COORDINATE COMPUTATION
    # =========================================================================
    
    async def compute_homogeneity(
        self,
        parent_content: str,
        child_content: str,
    ) -> float:
        """Compute semantic similarity between parent and child."""
        try:
            # Try to use embedding service
            from cognitive_twin._compat import EmbedderService
            import numpy as np
            
            embedder = EmbedderService()
            
            parent_emb = await embedder.embed(parent_content)
            child_emb = await embedder.embed(child_content)
            
            # Cosine similarity
            similarity = np.dot(parent_emb, child_emb) / (
                np.linalg.norm(parent_emb) * np.linalg.norm(child_emb)
            )
            
            return float(similarity)
        
        except (ImportError, Exception) as e:
            # Fallback: simple overlap
            parent_words = set(parent_content.lower().split())
            child_words = set(child_content.lower().split())
            
            if not parent_words or not child_words:
                return 0.0
            
            intersection = len(parent_words & child_words)
            union = len(parent_words | child_words)
            
            return intersection / union if union > 0 else 0.0
    
    # =========================================================================
    # RECORD CREATION
    # =========================================================================
    
    def create_sft_record(self, branch: SyntheticBranch) -> dict:
        """Create SFT record from synthetic branch."""
        return {
            "schema_version": "ctv3.1",
            "record_id": str(uuid4()),
            "record_type": "sft_turn",
            "source": {
                "origin": "convo_worm",
                "provider": "gpt-5.2",
                "source_id": branch.original_conversation_id,
                "created_at_utc": datetime.utcnow().isoformat(),
            },
            "context": {
                "domain": "mixed",
                "language": "en",
                "topology": {
                    "coords_5d": branch.coordinates.to_list(),
                    "phase_id": branch.phase_id,
                    "homogeneity": branch.coordinates.z,
                    "depth_norm": branch.coordinates.x / 10,
                    "sibling_order": branch.coordinates.y,
                    "temporal_norm": branch.coordinates.t,
                    "complexity": branch.coordinates.n,
                },
                "policy": {
                    "question_policy": branch.question_policy,
                    "directive_completeness": branch.directive_completeness,
                    "must_not_omit": False,
                    "format_constraints": {},
                },
            },
            "input": {
                "messages": branch.messages[:-1] if branch.messages else [],
                "attachments": [],
            },
            "target": {
                "assistant_content": branch.messages[-1]["content"] if branch.messages else "",
                "structured": {},
            },
            "tags": {
                "task_type": "respond",
                "prompt_class": "directive" if branch.directive_completeness >= 0.7 else "ambiguous",
                "branch_type": branch.branch_type,
            },
            "quality": {
                "gold": branch.is_gold,
                "weight": 1.0 if branch.is_gold else 0.5,
                "review_status": "auto",
                "failure_modes": [],
            },
        }
    
    def create_dpo_record(
        self,
        original_messages: list[dict],
        preferred_response: str,
        dispreferred_response: str,
        branch: SyntheticBranch,
    ) -> dict:
        """Create DPO pair record."""
        return {
            "schema_version": "ctv3.1",
            "record_id": str(uuid4()),
            "record_type": "dpo_pair",
            "source": {
                "origin": "convo_worm",
                "provider": "gpt-5.2",
                "source_id": branch.original_conversation_id,
                "created_at_utc": datetime.utcnow().isoformat(),
            },
            "context": {
                "domain": "mixed",
                "language": "en",
                "topology": {
                    "coords_5d": branch.coordinates.to_list(),
                    "phase_id": branch.phase_id,
                },
                "policy": {
                    "question_policy": "no_questions",
                    "directive_completeness": branch.directive_completeness,
                },
            },
            "input": {
                "messages": original_messages,
                "attachments": [],
            },
            "candidates": {
                "preferred": {"assistant_content": preferred_response},
                "dispreferred": {"assistant_content": dispreferred_response},
            },
            "tags": {
                "task_type": "respond",
                "prompt_class": "directive",
                "dpo_reason": "friction_repair",
            },
            "quality": {
                "gold": True,
                "weight": 1.0,
                "review_status": "auto",
                "failure_modes": [],
            },
        }
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_sft(
        self,
        records: list[dict],
        output_path: Path,
    ):
        """Export SFT records to JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Exported {len(records)} SFT records to {output_path}")
    
    def export_dpo(
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


class ConversationWormPipeline:
    """Complete Conversation Worm pipeline."""
    
    def __init__(
        self,
        supabase_client=None,
        config: Optional[ConversationWormConfig] = None,
    ):
        self.supabase = supabase_client
        self.config = config or ConversationWormConfig()
    
    async def run(
        self,
        conversation_ids: Optional[list[str]] = None,
        output_dir: Optional[Path] = None,
        max_conversations: int = 100,
    ) -> ConversationWormStats:
        """Run the complete pipeline."""
        stats = ConversationWormStats()
        stats.start_time = datetime.utcnow().isoformat()
        
        # Initialize worm
        worm = ConversationWorm(self.supabase, self.config)
        
        # Get conversations to process
        if conversation_ids is None:
            conversation_ids = await worm.get_all_conversation_ids(max_conversations)
        else:
            conversation_ids = conversation_ids[:max_conversations]
        
        if not conversation_ids:
            logger.warning("No conversations to process")
            stats.end_time = datetime.utcnow().isoformat()
            return stats
        
        logger.info(f"Processing {len(conversation_ids)} conversations")
        
        # Process conversations
        results = await worm.process_batch(conversation_ids)
        
        # Collect all branches
        sft_records = []
        dpo_records = []
        
        for result in results:
            if not result.success:
                continue
            
            stats.conversations_processed += 1
            stats.friction_points_found += result.friction_points_found
            stats.paraphrase_branches += len(result.paraphrase_branches)
            stats.ideal_branches += len(result.ideal_branches)
            stats.extension_branches += len(result.extension_branches)
            stats.contrast_branches += len(result.contrast_branches)
            
            for branch in result.all_branches:
                # Create SFT record
                sft = worm.create_sft_record(branch)
                sft_records.append(sft)
                
                # Create DPO pair if it's an ideal response
                if branch.branch_type == "ideal_response":
                    # Find original dispreferred response
                    original = self._get_original_response(branch)
                    
                    if original and branch.messages:
                        dpo = worm.create_dpo_record(
                            branch.messages[:-1],
                            branch.messages[-1]["content"],  # Preferred
                            original,  # Dispreferred
                            branch,
                        )
                        dpo_records.append(dpo)
        
        stats.sft_records = len(sft_records)
        stats.dpo_pairs = len(dpo_records)
        
        # Export
        if output_dir:
            output_dir = Path(output_dir)
            worm.export_sft(sft_records, output_dir / "convo_sft.jsonl")
            worm.export_dpo(dpo_records, output_dir / "convo_dpo.jsonl")
            
            # Write stats
            with open(output_dir / "convo_stats.json", 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)
        
        stats.end_time = datetime.utcnow().isoformat()
        
        if stats.start_time and stats.end_time:
            start = datetime.fromisoformat(stats.start_time)
            end = datetime.fromisoformat(stats.end_time)
            stats.duration_seconds = (end - start).total_seconds()
        
        logger.info(f"Pipeline complete: {stats.to_dict()}")
        
        return stats
    
    def _get_original_response(self, branch: SyntheticBranch) -> Optional[str]:
        """Get the original dispreferred response for a friction branch."""
        # This would typically look up the original response from Supabase
        # For now, return None as we don't have direct access
        return None


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Conversation Worm for CognitiveTwin V3"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/convo_worm_output"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-conversations", "-n",
        type=int,
        default=100,
        help="Maximum conversations to process",
    )
    parser.add_argument(
        "--conversation-ids",
        nargs="*",
        help="Specific conversation IDs to process",
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
    pipeline = ConversationWormPipeline(supabase)
    
    stats = await pipeline.run(
        conversation_ids=args.conversation_ids,
        output_dir=args.output,
        max_conversations=args.max_conversations,
    )
    
    print("\n" + "=" * 60)
    print("Conversation Worm Complete")
    print("=" * 60)
    print(f"Conversations processed: {stats.conversations_processed}")
    print(f"Friction points found: {stats.friction_points_found}")
    print(f"Total branches: {stats.total_branches}")
    print(f"  - Paraphrases: {stats.paraphrase_branches}")
    print(f"  - Ideal responses: {stats.ideal_branches}")
    print(f"  - Extensions: {stats.extension_branches}")
    print(f"  - Contrast pairs: {stats.contrast_branches}")
    print(f"SFT records: {stats.sft_records}")
    print(f"DPO pairs: {stats.dpo_pairs}")
    print(f"Duration: {stats.duration_seconds:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

