"""
DPO Pair Factory - Creates DPO training pairs using V2 as preferred generator.

Uses the fine-tuned V2 model to generate style-aligned preferred responses,
and the base model to generate dispreferred responses with problematic patterns.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .v2_generator import V2Generator, GeneratorConfig, GenerationResult

logger = logging.getLogger(__name__)


class DPOPairType(str, Enum):
    """Types of DPO pairs for different failure modes."""
    CONFIRMATION_REFLEX = "confirmation_reflex"
    OPTION_SPAM = "option_spam"
    OMISSION = "omission"
    FORMAT_DRIFT = "format_drift"
    PERMISSION_SEEKING = "permission_seeking"
    EXCESSIVE_HEDGING = "excessive_hedging"
    REWRITE = "rewrite"  # From corpus surgery
    BRANCH = "branch"    # From conversation worm


@dataclass
class DPOPair:
    """A single DPO training pair."""
    prompt: str
    chosen: str  # Preferred response (from V2)
    rejected: str  # Dispreferred response (from base)
    
    # Metadata
    pair_type: DPOPairType = DPOPairType.PERMISSION_SEEKING
    source_conversation_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Quality metrics
    chosen_tokens: int = 0
    rejected_tokens: int = 0
    length_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL export."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
    
    def to_full_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all metadata."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "pair_type": self.pair_type.value,
            "source_conversation_id": self.source_conversation_id,
            "created_at": self.created_at,
            "chosen_tokens": self.chosen_tokens,
            "rejected_tokens": self.rejected_tokens,
            "length_ratio": self.length_ratio,
        }


@dataclass
class FactoryStats:
    """Statistics from DPO pair generation."""
    total_prompts: int = 0
    successful_pairs: int = 0
    failed_pairs: int = 0
    pairs_by_type: Dict[str, int] = field(default_factory=dict)
    total_tokens_used: int = 0
    avg_latency_ms: float = 0.0


class DPOFactory:
    """
    Factory for creating DPO training pairs using V2 as preferred generator.
    
    The key insight is that V2 has already learned from user corrections,
    so it naturally produces preferred-style responses. The base model
    produces the "bad" patterns we want to train away.
    """
    
    def __init__(
        self,
        generator: Optional[V2Generator] = None,
        config: Optional[GeneratorConfig] = None,
    ):
        self.generator = generator or V2Generator(config)
        self.stats = FactoryStats()
        self._pairs: List[DPOPair] = []
    
    async def create_confirmation_reflex_pair(
        self,
        prompt: str,
        source_id: Optional[str] = None,
    ) -> Optional[DPOPair]:
        """
        Create a pair where V2 executes directly vs base asks for confirmation.
        
        Args:
            prompt: The user request
            source_id: Optional source conversation ID
            
        Returns:
            DPOPair or None if generation failed
        """
        preferred = await self.generator.generate_preferred(prompt)
        dispreferred = await self.generator.generate_dispreferred(
            prompt, failure_mode="permission_seeking"
        )
        
        if not preferred.success or not dispreferred.success:
            self.stats.failed_pairs += 1
            return None
        
        pair = DPOPair(
            prompt=prompt,
            chosen=preferred.content,
            rejected=dispreferred.content,
            pair_type=DPOPairType.CONFIRMATION_REFLEX,
            source_conversation_id=source_id,
            chosen_tokens=preferred.tokens_used,
            rejected_tokens=dispreferred.tokens_used,
            length_ratio=len(preferred.content) / max(len(dispreferred.content), 1),
        )
        
        self._pairs.append(pair)
        self.stats.successful_pairs += 1
        self.stats.pairs_by_type["confirmation_reflex"] = \
            self.stats.pairs_by_type.get("confirmation_reflex", 0) + 1
        
        return pair
    
    async def create_option_spam_pair(
        self,
        prompt: str,
        source_id: Optional[str] = None,
    ) -> Optional[DPOPair]:
        """
        Create a pair where V2 gives single best choice vs base lists options.
        """
        # Modify prompt to encourage single choice from V2
        v2_prompt = f"{prompt}\n\nProvide the single best solution."
        
        preferred = await self.generator.generate_preferred(v2_prompt)
        dispreferred = await self.generator.generate_dispreferred(
            prompt, failure_mode="option_spam"
        )
        
        if not preferred.success or not dispreferred.success:
            self.stats.failed_pairs += 1
            return None
        
        pair = DPOPair(
            prompt=prompt,
            chosen=preferred.content,
            rejected=dispreferred.content,
            pair_type=DPOPairType.OPTION_SPAM,
            source_conversation_id=source_id,
            chosen_tokens=preferred.tokens_used,
            rejected_tokens=dispreferred.tokens_used,
        )
        
        self._pairs.append(pair)
        self.stats.successful_pairs += 1
        self.stats.pairs_by_type["option_spam"] = \
            self.stats.pairs_by_type.get("option_spam", 0) + 1
        
        return pair
    
    async def create_omission_pair(
        self,
        prompt: str,
        source_id: Optional[str] = None,
    ) -> Optional[DPOPair]:
        """
        Create a pair where V2 gives complete output vs base omits content.
        """
        preferred = await self.generator.generate_preferred(prompt)
        dispreferred = await self.generator.generate_dispreferred(
            prompt, failure_mode="incomplete"
        )
        
        if not preferred.success or not dispreferred.success:
            self.stats.failed_pairs += 1
            return None
        
        pair = DPOPair(
            prompt=prompt,
            chosen=preferred.content,
            rejected=dispreferred.content,
            pair_type=DPOPairType.OMISSION,
            source_conversation_id=source_id,
            chosen_tokens=preferred.tokens_used,
            rejected_tokens=dispreferred.tokens_used,
        )
        
        self._pairs.append(pair)
        self.stats.successful_pairs += 1
        self.stats.pairs_by_type["omission"] = \
            self.stats.pairs_by_type.get("omission", 0) + 1
        
        return pair
    
    async def create_rewrite_pair(
        self,
        user_message: str,
        bad_assistant_message: str,
        source_id: Optional[str] = None,
    ) -> Optional[DPOPair]:
        """
        Create a pair from corpus surgery rewriting.
        
        V2 rewrites the unjustified turn into proper execution.
        The original bad message becomes the rejected response.
        
        Args:
            user_message: The original user request
            bad_assistant_message: The unjustified assistant response
            source_id: Optional source conversation ID
            
        Returns:
            DPOPair with V2's rewrite as chosen
        """
        rewritten = await self.generator.rewrite_turn(
            user_message=user_message,
            bad_assistant_message=bad_assistant_message,
        )
        
        if not rewritten.success:
            self.stats.failed_pairs += 1
            return None
        
        pair = DPOPair(
            prompt=user_message,
            chosen=rewritten.content,
            rejected=bad_assistant_message,
            pair_type=DPOPairType.REWRITE,
            source_conversation_id=source_id,
            chosen_tokens=rewritten.tokens_used,
        )
        
        self._pairs.append(pair)
        self.stats.successful_pairs += 1
        self.stats.pairs_by_type["rewrite"] = \
            self.stats.pairs_by_type.get("rewrite", 0) + 1
        
        return pair
    
    async def create_branch_pair(
        self,
        prompt: str,
        context: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> Optional[DPOPair]:
        """
        Create a pair for conversation branching.
        
        V2 generates ideal response, base generates hedging response.
        """
        if context:
            full_prompt = f"Context:\n{context}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
        
        preferred = await self.generator.generate_ideal_response(prompt, context)
        dispreferred = await self.generator.generate_dispreferred(
            full_prompt, failure_mode="excessive_hedging"
        )
        
        if not preferred.success or not dispreferred.success:
            self.stats.failed_pairs += 1
            return None
        
        pair = DPOPair(
            prompt=prompt,
            chosen=preferred.content,
            rejected=dispreferred.content,
            pair_type=DPOPairType.BRANCH,
            source_conversation_id=source_id,
            chosen_tokens=preferred.tokens_used,
            rejected_tokens=dispreferred.tokens_used,
        )
        
        self._pairs.append(pair)
        self.stats.successful_pairs += 1
        self.stats.pairs_by_type["branch"] = \
            self.stats.pairs_by_type.get("branch", 0) + 1
        
        return pair
    
    async def create_pairs_for_conversation(
        self,
        conversation: Dict[str, Any],
        include_types: Optional[List[DPOPairType]] = None,
    ) -> List[DPOPair]:
        """
        Create multiple DPO pairs from a single conversation.
        
        Args:
            conversation: Unified conversation dict
            include_types: Types of pairs to generate (all if None)
            
        Returns:
            List of generated DPO pairs
        """
        pairs = []
        conv_id = conversation.get("conversation_id", "unknown")
        turns = conversation.get("turns", [])
        
        # Default to all types
        if include_types is None:
            include_types = [
                DPOPairType.CONFIRMATION_REFLEX,
                DPOPairType.OPTION_SPAM,
            ]
        
        # Process user turns
        for i, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue
            
            prompt = turn.get("content", "")
            if not prompt or len(prompt) < 20:
                continue
            
            self.stats.total_prompts += 1
            
            # Generate pairs for each requested type
            for pair_type in include_types:
                try:
                    if pair_type == DPOPairType.CONFIRMATION_REFLEX:
                        pair = await self.create_confirmation_reflex_pair(prompt, conv_id)
                    elif pair_type == DPOPairType.OPTION_SPAM:
                        pair = await self.create_option_spam_pair(prompt, conv_id)
                    elif pair_type == DPOPairType.OMISSION:
                        pair = await self.create_omission_pair(prompt, conv_id)
                    else:
                        pair = await self.create_confirmation_reflex_pair(prompt, conv_id)
                    
                    if pair:
                        pairs.append(pair)
                        
                except Exception as e:
                    logger.warning(f"Failed to create pair: {e}")
                    self.stats.failed_pairs += 1
        
        return pairs
    
    async def batch_create_pairs(
        self,
        prompts: List[str],
        pair_type: DPOPairType = DPOPairType.CONFIRMATION_REFLEX,
    ) -> List[DPOPair]:
        """
        Create pairs for multiple prompts.
        """
        pairs = []
        
        for prompt in prompts:
            self.stats.total_prompts += 1
            
            try:
                if pair_type == DPOPairType.CONFIRMATION_REFLEX:
                    pair = await self.create_confirmation_reflex_pair(prompt)
                elif pair_type == DPOPairType.OPTION_SPAM:
                    pair = await self.create_option_spam_pair(prompt)
                elif pair_type == DPOPairType.OMISSION:
                    pair = await self.create_omission_pair(prompt)
                else:
                    pair = await self.create_confirmation_reflex_pair(prompt)
                
                if pair:
                    pairs.append(pair)
                    
            except Exception as e:
                logger.warning(f"Failed to create pair for prompt: {e}")
                self.stats.failed_pairs += 1
        
        return pairs
    
    def get_all_pairs(self) -> List[DPOPair]:
        """Get all generated pairs."""
        return self._pairs
    
    def export_jsonl(self, path: Path) -> int:
        """
        Export pairs to JSONL file.
        
        Args:
            path: Output file path
            
        Returns:
            Number of pairs exported
        """
        with open(path, "w") as f:
            for pair in self._pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")
        
        logger.info(f"Exported {len(self._pairs)} DPO pairs to {path}")
        return len(self._pairs)
    
    def export_with_metadata(self, path: Path) -> int:
        """Export pairs with full metadata."""
        with open(path, "w") as f:
            for pair in self._pairs:
                f.write(json.dumps(pair.to_full_dict()) + "\n")
        
        return len(self._pairs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_prompts": self.stats.total_prompts,
            "successful_pairs": self.stats.successful_pairs,
            "failed_pairs": self.stats.failed_pairs,
            "success_rate": self.stats.successful_pairs / max(self.stats.total_prompts, 1),
            "pairs_by_type": self.stats.pairs_by_type,
            "total_pairs": len(self._pairs),
        }


# Convenience function
async def generate_dpo_pairs(
    prompts: List[str],
    pair_type: DPOPairType = DPOPairType.CONFIRMATION_REFLEX,
    v2_model: Optional[str] = None,
) -> List[DPOPair]:
    """
    Quick DPO pair generation.
    
    Args:
        prompts: List of user prompts
        pair_type: Type of pairs to generate
        v2_model: Optional V2 model override
        
    Returns:
        List of generated DPO pairs
    """
    config = GeneratorConfig()
    if v2_model:
        config.v2_model = v2_model
    
    factory = DPOFactory(config=config)
    return await factory.batch_create_pairs(prompts, pair_type)

