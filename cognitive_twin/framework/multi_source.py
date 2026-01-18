"""Multi-Source Fusion for CognitiveTwin.

This module enables training CognitiveTwin on multiple data sources
simultaneously, learning unified representations across:

1. Prompts/Responses (conversation data)
2. Notes (Obsidian, Apple Notes)
3. Code (files, commits)
4. Motion (Echelon rehearsal data)
5. Images (with captions/descriptions)
6. Social (posts, interactions)

Core Concepts:
    - Source Encoders: Modality-specific encoders
    - Fusion Layer: Combines multi-source embeddings
    - Cross-Modal Contrastive: Align representations across modalities
    - Source-Aware Attention: Learn modality-specific patterns

Usage:
    >>> from cognitive_twin.framework.multi_source import MultiSourceFusion
    >>> fusion = MultiSourceFusion(embed_dim=768)
    >>> 
    >>> # Process different sources
    >>> unified = fusion(
    ...     prompt_embeddings=prompt_embs,
    ...     note_embeddings=note_embs,
    ...     code_embeddings=code_embs,
    ... )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# Source Types and Configuration
# =============================================================================

class SourceType(str, Enum):
    """Types of data sources."""
    
    PROMPT = "prompt"
    NOTE = "note"
    CODE = "code"
    MOTION = "motion"
    IMAGE = "image"
    SOCIAL = "social"
    DOCUMENT = "document"


@dataclass
class MultiSourceConfig:
    """Configuration for multi-source fusion.
    
    Attributes:
        embed_dim: Unified embedding dimension.
        num_sources: Number of source types.
        fusion_type: Type of fusion (concat, attention, gated).
        hidden_dim: Hidden layer dimension.
        num_heads: Attention heads for fusion.
        dropout: Dropout rate.
        use_source_embeddings: Learn source-specific embeddings.
        cross_modal_weight: Weight for cross-modal contrastive.
        alignment_weight: Weight for modality alignment loss.
        source_weights: Optional per-source weights.
    """
    
    embed_dim: int = 768
    num_sources: int = 6
    fusion_type: str = "attention"  # concat, attention, gated
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    use_source_embeddings: bool = True
    cross_modal_weight: float = 0.2
    alignment_weight: float = 0.1
    source_weights: Optional[Dict[SourceType, float]] = None
    
    def get_source_weight(self, source: SourceType) -> float:
        """Get weight for a source type."""
        if self.source_weights:
            return self.source_weights.get(source, 1.0)
        return 1.0


@dataclass
class SourceBatch:
    """Batch of data from a single source.
    
    Attributes:
        source_type: Type of data source.
        embeddings: Embeddings [batch, dim].
        mask: Valid mask [batch].
        coordinates: Optional trajectory coordinates [batch, 5].
        metadata: Additional metadata.
    """
    
    source_type: SourceType
    embeddings: Tensor
    mask: Optional[Tensor] = None
    coordinates: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def batch_size(self) -> int:
        return self.embeddings.shape[0]


@dataclass
class FusedOutput:
    """Output from multi-source fusion.
    
    Attributes:
        unified_embedding: Fused embedding [batch, dim].
        source_weights: Learned source weights [batch, num_sources].
        per_source_embeddings: Per-source processed embeddings.
        alignment_loss: Cross-modal alignment loss.
        attention_weights: Fusion attention weights.
    """
    
    unified_embedding: Tensor
    source_weights: Optional[Tensor] = None
    per_source_embeddings: Optional[Dict[SourceType, Tensor]] = None
    alignment_loss: Optional[Tensor] = None
    attention_weights: Optional[Tensor] = None


# =============================================================================
# Source Encoder
# =============================================================================

class SourceEncoder(nn.Module):
    """Modality-specific encoder for a data source.
    
    Projects source embeddings into a shared space with source-specific
    transformations.
    """
    
    def __init__(
        self,
        source_type: SourceType,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize source encoder.
        
        Args:
            source_type: Type of data source.
            input_dim: Input embedding dimension.
            output_dim: Output (unified) dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.source_type = source_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.hidden = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Layer norm for output
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, embeddings: Tensor) -> Tensor:
        """Encode source embeddings.
        
        Args:
            embeddings: Source embeddings [batch, input_dim].
            
        Returns:
            Encoded embeddings [batch, output_dim].
        """
        x = self.input_proj(embeddings)
        x = F.gelu(x)
        x = self.hidden(x)
        x = self.output_proj(x)
        x = self.output_norm(x)
        return x


# =============================================================================
# Fusion Layers
# =============================================================================

class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple sources.
    
    Uses multi-head attention to learn how to combine embeddings
    from different sources based on context.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize attention fusion.
        
        Args:
            embed_dim: Embedding dimension.
            num_sources: Maximum number of sources.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.num_heads = num_heads
        
        # Source embeddings
        self.source_embeddings = nn.Embedding(num_sources, embed_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        source_embeddings: Dict[SourceType, Tensor],
        source_indices: Dict[SourceType, int],
    ) -> Tuple[Tensor, Tensor]:
        """Fuse source embeddings with attention.
        
        Args:
            source_embeddings: Dict of source_type -> embeddings [batch, dim].
            source_indices: Dict of source_type -> index for embedding lookup.
            
        Returns:
            Tuple of (fused_embedding, attention_weights).
        """
        # Get batch size from first source
        first_source = next(iter(source_embeddings.values()))
        batch_size = first_source.shape[0]
        device = first_source.device
        
        # Stack all source embeddings [batch, num_active_sources, dim]
        active_sources = []
        source_idx_list = []
        
        for source_type, emb in source_embeddings.items():
            idx = source_indices.get(source_type, 0)
            source_idx_list.append(idx)
            # Add source embedding
            source_emb = self.source_embeddings(
                torch.tensor([idx], device=device)
            ).expand(batch_size, -1)
            combined = emb + source_emb
            active_sources.append(combined)
        
        if not active_sources:
            return torch.zeros(batch_size, self.embed_dim, device=device), None
        
        # Stack sources [batch, num_sources, dim]
        stacked = torch.stack(active_sources, dim=1)
        
        # Create query as mean of all sources
        query = stacked.mean(dim=1, keepdim=True)
        
        # Attention
        fused, attn_weights = self.attention(
            query=query,
            key=stacked,
            value=stacked,
        )
        
        # Project and normalize
        fused = fused.squeeze(1)
        fused = self.output_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused, attn_weights


class GatedFusion(nn.Module):
    """Gated fusion of multiple sources.
    
    Learns a gating mechanism to combine sources based on
    their content and relevance.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        hidden_dim: int = 256,
    ):
        """Initialize gated fusion.
        
        Args:
            embed_dim: Embedding dimension.
            num_sources: Maximum number of sources.
            hidden_dim: Hidden dimension for gate.
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_sources = num_sources
        
        # Gate network
        self.gate_input = nn.Linear(embed_dim * num_sources, hidden_dim)
        self.gate_output = nn.Linear(hidden_dim, num_sources)
        
        # Combine projection
        self.combine = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        source_embeddings: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Fuse source embeddings with gating.
        
        Args:
            source_embeddings: List of source embeddings [batch, dim].
            
        Returns:
            Tuple of (fused_embedding, gate_weights).
        """
        batch_size = source_embeddings[0].shape[0]
        device = source_embeddings[0].device
        
        # Pad to num_sources
        padded = []
        for i in range(self.num_sources):
            if i < len(source_embeddings):
                padded.append(source_embeddings[i])
            else:
                padded.append(torch.zeros(
                    batch_size, self.embed_dim, device=device
                ))
        
        # Stack and flatten for gate
        stacked = torch.stack(padded, dim=1)  # [batch, num_sources, dim]
        flat = stacked.view(batch_size, -1)  # [batch, num_sources * dim]
        
        # Compute gate
        gate = self.gate_input(flat)
        gate = F.relu(gate)
        gate = self.gate_output(gate)
        gate = F.softmax(gate, dim=-1)  # [batch, num_sources]
        
        # Apply gate
        gate_expanded = gate.unsqueeze(-1)  # [batch, num_sources, 1]
        weighted = stacked * gate_expanded
        fused = weighted.sum(dim=1)  # [batch, dim]
        
        # Project and normalize
        fused = self.combine(fused)
        fused = self.layer_norm(fused)
        
        return fused, gate


# =============================================================================
# Cross-Modal Contrastive Loss
# =============================================================================

class CrossModalContrastiveLoss(nn.Module):
    """Contrastive loss for aligning different modalities.
    
    Learns to align embeddings from different sources that represent
    the same concept or are semantically related.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        alignment_type: str = "symmetric",  # symmetric, asymmetric
    ):
        """Initialize cross-modal contrastive loss.
        
        Args:
            temperature: Temperature for softmax scaling.
            alignment_type: Type of alignment loss.
        """
        super().__init__()
        self.temperature = temperature
        self.alignment_type = alignment_type
        
    def forward(
        self,
        source_a: Tensor,
        source_b: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute cross-modal contrastive loss.
        
        Args:
            source_a: Embeddings from source A [batch, dim].
            source_b: Embeddings from source B [batch, dim].
            labels: Optional alignment labels [batch] (same = should align).
            
        Returns:
            Tuple of (loss, metrics).
        """
        batch_size = source_a.shape[0]
        device = source_a.device
        
        # Normalize
        source_a = F.normalize(source_a, p=2, dim=-1)
        source_b = F.normalize(source_b, p=2, dim=-1)
        
        # Compute similarity
        sim_a_to_b = torch.matmul(source_a, source_b.T) / self.temperature
        sim_b_to_a = sim_a_to_b.T
        
        # Labels: diagonal = positive (same sample index)
        if labels is None:
            labels = torch.arange(batch_size, device=device)
        
        # Cross-entropy loss (matching to same index)
        loss_a_to_b = F.cross_entropy(sim_a_to_b, labels)
        loss_b_to_a = F.cross_entropy(sim_b_to_a, labels)
        
        if self.alignment_type == "symmetric":
            loss = (loss_a_to_b + loss_b_to_a) / 2
        else:
            loss = loss_a_to_b
        
        # Metrics
        with torch.no_grad():
            # Accuracy
            pred_a_to_b = sim_a_to_b.argmax(dim=-1)
            pred_b_to_a = sim_b_to_a.argmax(dim=-1)
            acc_a_to_b = (pred_a_to_b == labels).float().mean()
            acc_b_to_a = (pred_b_to_a == labels).float().mean()
            
            # Average similarity of positives
            positive_sim = torch.diag(sim_a_to_b).mean()
        
        metrics = {
            "cross_modal_loss": loss,
            "acc_a_to_b": acc_a_to_b,
            "acc_b_to_a": acc_b_to_a,
            "positive_similarity": positive_sim,
        }
        
        return loss, metrics


# =============================================================================
# Multi-Source Fusion
# =============================================================================

class MultiSourceFusion(nn.Module):
    """Multi-source fusion module for CognitiveTwin.
    
    Combines embeddings from multiple data sources into a unified
    representation for training and inference.
    """
    
    def __init__(self, config: MultiSourceConfig):
        """Initialize multi-source fusion.
        
        Args:
            config: Fusion configuration.
        """
        super().__init__()
        self.config = config
        
        # Source type to index mapping
        self.source_to_idx = {
            SourceType.PROMPT: 0,
            SourceType.NOTE: 1,
            SourceType.CODE: 2,
            SourceType.MOTION: 3,
            SourceType.IMAGE: 4,
            SourceType.SOCIAL: 5,
        }
        
        # Source-specific encoders
        self.source_encoders = nn.ModuleDict()
        for source_type in SourceType:
            self.source_encoders[source_type.value] = SourceEncoder(
                source_type=source_type,
                input_dim=config.embed_dim,
                output_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        
        # Fusion layer
        if config.fusion_type == "attention":
            self.fusion = AttentionFusion(
                embed_dim=config.embed_dim,
                num_sources=config.num_sources,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        elif config.fusion_type == "gated":
            self.fusion = GatedFusion(
                embed_dim=config.embed_dim,
                num_sources=config.num_sources,
                hidden_dim=config.hidden_dim,
            )
        else:  # concat
            self.fusion = nn.Linear(
                config.embed_dim * config.num_sources,
                config.embed_dim,
            )
        
        # Cross-modal contrastive
        self.cross_modal_loss = CrossModalContrastiveLoss()
        
        # Source embeddings for conditioning
        if config.use_source_embeddings:
            self.source_embeddings = nn.Embedding(
                config.num_sources, config.embed_dim
            )
        else:
            self.source_embeddings = None
        
        logger.info(f"MultiSourceFusion initialized: fusion={config.fusion_type}")
    
    def forward(
        self,
        sources: Dict[SourceType, Tensor],
        compute_alignment: bool = True,
    ) -> FusedOutput:
        """Fuse embeddings from multiple sources.
        
        Args:
            sources: Dict of source_type -> embeddings [batch, dim].
            compute_alignment: Whether to compute alignment loss.
            
        Returns:
            FusedOutput with unified embedding.
        """
        if not sources:
            raise ValueError("At least one source must be provided")
        
        # Get batch size and device
        first_source = next(iter(sources.values()))
        batch_size = first_source.shape[0]
        device = first_source.device
        
        # Encode each source
        encoded_sources = {}
        for source_type, embeddings in sources.items():
            encoder = self.source_encoders[source_type.value]
            encoded = encoder(embeddings)
            
            # Add source embedding if available
            if self.source_embeddings is not None:
                idx = self.source_to_idx[source_type]
                source_emb = self.source_embeddings(
                    torch.tensor([idx], device=device)
                ).expand(batch_size, -1)
                encoded = encoded + source_emb
            
            encoded_sources[source_type] = encoded
        
        # Apply fusion
        if self.config.fusion_type == "attention":
            fused, attn_weights = self.fusion(
                encoded_sources, self.source_to_idx
            )
            source_weights = attn_weights
        elif self.config.fusion_type == "gated":
            # Convert to list
            encoded_list = [
                encoded_sources.get(st, torch.zeros(batch_size, self.config.embed_dim, device=device))
                for st in SourceType
            ]
            fused, gate_weights = self.fusion(encoded_list)
            source_weights = gate_weights
            attn_weights = None
        else:  # concat
            # Pad and concatenate
            padded = []
            for st in SourceType:
                if st in encoded_sources:
                    padded.append(encoded_sources[st])
                else:
                    padded.append(torch.zeros(
                        batch_size, self.config.embed_dim, device=device
                    ))
            concat = torch.cat(padded, dim=-1)
            fused = self.fusion(concat)
            source_weights = None
            attn_weights = None
        
        # Compute alignment loss
        alignment_loss = None
        if compute_alignment and len(encoded_sources) >= 2:
            alignment_loss = self._compute_alignment_loss(encoded_sources)
        
        return FusedOutput(
            unified_embedding=fused,
            source_weights=source_weights,
            per_source_embeddings=encoded_sources,
            alignment_loss=alignment_loss,
            attention_weights=attn_weights,
        )
    
    def _compute_alignment_loss(
        self,
        encoded_sources: Dict[SourceType, Tensor],
    ) -> Tensor:
        """Compute cross-modal alignment loss.
        
        Args:
            encoded_sources: Dict of encoded source embeddings.
            
        Returns:
            Alignment loss tensor.
        """
        source_list = list(encoded_sources.values())
        
        if len(source_list) < 2:
            return torch.tensor(0.0, device=source_list[0].device)
        
        total_loss = torch.tensor(0.0, device=source_list[0].device)
        num_pairs = 0
        
        # Pairwise alignment loss
        for i in range(len(source_list)):
            for j in range(i + 1, len(source_list)):
                loss, _ = self.cross_modal_loss(
                    source_list[i], source_list[j]
                )
                total_loss = total_loss + loss
                num_pairs += 1
        
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return total_loss
    
    def encode_source(
        self,
        source_type: SourceType,
        embeddings: Tensor,
    ) -> Tensor:
        """Encode a single source.
        
        Args:
            source_type: Type of source.
            embeddings: Source embeddings [batch, dim].
            
        Returns:
            Encoded embeddings [batch, dim].
        """
        encoder = self.source_encoders[source_type.value]
        encoded = encoder(embeddings)
        
        if self.source_embeddings is not None:
            idx = self.source_to_idx[source_type]
            batch_size = embeddings.shape[0]
            device = embeddings.device
            source_emb = self.source_embeddings(
                torch.tensor([idx], device=device)
            ).expand(batch_size, -1)
            encoded = encoded + source_emb
        
        return encoded


# =============================================================================
# Multi-Source Dataset
# =============================================================================

class MultiSourceDataset(torch.utils.data.Dataset):
    """Dataset for multi-source training.
    
    Groups samples from different sources for joint training.
    """
    
    def __init__(
        self,
        sources: Dict[SourceType, List[Dict[str, Any]]],
        embed_dim: int = 768,
    ):
        """Initialize dataset.
        
        Args:
            sources: Dict of source_type -> list of samples.
            embed_dim: Embedding dimension.
        """
        self.sources = sources
        self.embed_dim = embed_dim
        
        # Build index of all samples
        self.samples = []
        for source_type, samples in sources.items():
            for sample in samples:
                self.samples.append({
                    "source_type": source_type,
                    **sample,
                })
        
        logger.info(
            f"MultiSourceDataset: {len(self.samples)} samples from "
            f"{len(sources)} sources"
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
    
    @staticmethod
    def collate_fn(
        samples: List[Dict[str, Any]],
        embed_dim: int = 768,
    ) -> Dict[SourceType, SourceBatch]:
        """Collate samples into source batches.
        
        Args:
            samples: List of sample dicts.
            embed_dim: Embedding dimension.
            
        Returns:
            Dict of source_type -> SourceBatch.
        """
        # Group by source type
        by_source: Dict[SourceType, List[Dict]] = {}
        for sample in samples:
            source_type = sample["source_type"]
            if source_type not in by_source:
                by_source[source_type] = []
            by_source[source_type].append(sample)
        
        # Create batches
        batches = {}
        for source_type, source_samples in by_source.items():
            embeddings = []
            coords = []
            
            for sample in source_samples:
                emb = sample.get("embedding")
                if emb is None:
                    emb = torch.zeros(embed_dim)
                elif isinstance(emb, list):
                    emb = torch.tensor(emb)
                embeddings.append(emb)
                
                coord = sample.get("trajectory_coords")
                if coord is not None:
                    if isinstance(coord, list):
                        coord = torch.tensor(coord)
                    coords.append(coord)
            
            batches[source_type] = SourceBatch(
                source_type=source_type,
                embeddings=torch.stack(embeddings),
                coordinates=torch.stack(coords) if coords else None,
            )
        
        return batches


# =============================================================================
# Factory Function
# =============================================================================

def create_multi_source_fusion(
    embed_dim: int = 768,
    fusion_type: str = "attention",
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs,
) -> MultiSourceFusion:
    """Create a multi-source fusion module.
    
    Args:
        embed_dim: Embedding dimension.
        fusion_type: Type of fusion (attention, gated, concat).
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        **kwargs: Additional config options.
        
    Returns:
        Configured MultiSourceFusion.
    """
    config = MultiSourceConfig(
        embed_dim=embed_dim,
        fusion_type=fusion_type,
        num_heads=num_heads,
        dropout=dropout,
        **kwargs,
    )
    
    return MultiSourceFusion(config)

