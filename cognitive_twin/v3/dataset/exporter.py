"""
Dataset Exporter for CognitiveTwin V3.

Exports datasets with:
- Multiple formats (JSONL, Parquet)
- Train/val/test splits
- Quality filtering
"""

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from ..schema import (
    CTv3Record,
    CTv3DPORecord,
    CTv3EvalRecord,
)


logger = logging.getLogger(__name__)


# =============================================================================
# EXPORT FORMAT
# =============================================================================

class ExportFormat(str, Enum):
    """Export formats."""
    JSONL = "jsonl"
    PARQUET = "parquet"
    CSV = "csv"


# =============================================================================
# DATASET SPLIT
# =============================================================================

@dataclass
class DatasetSplit:
    """Dataset split configuration."""
    
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    @classmethod
    def from_string(cls, s: str) -> "DatasetSplit":
        """Parse from string like '0.8,0.1,0.1'."""
        parts = [float(x.strip()) for x in s.split(',')]
        if len(parts) != 3:
            raise ValueError(f"Expected 3 values, got {len(parts)}")
        return cls(train_ratio=parts[0], val_ratio=parts[1], test_ratio=parts[2])


# =============================================================================
# DATASET EXPORTER
# =============================================================================

class DatasetExporter:
    """Export CTv3 records to files."""
    
    def export_sft(
        self,
        records: list[CTv3Record],
        output_path: Path,
        format: ExportFormat = ExportFormat.JSONL,
    ) -> int:
        """Export SFT records."""
        if format == ExportFormat.JSONL:
            return self._export_jsonl(records, output_path)
        elif format == ExportFormat.PARQUET:
            return self._export_parquet(records, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_dpo(
        self,
        records: list[CTv3DPORecord],
        output_path: Path,
        format: ExportFormat = ExportFormat.JSONL,
    ) -> int:
        """Export DPO records."""
        if format == ExportFormat.JSONL:
            return self._export_jsonl(records, output_path)
        elif format == ExportFormat.PARQUET:
            return self._export_parquet(records, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_eval(
        self,
        records: list[CTv3EvalRecord],
        output_path: Path,
        format: ExportFormat = ExportFormat.JSONL,
    ) -> int:
        """Export eval records."""
        if format == ExportFormat.JSONL:
            return self._export_jsonl(records, output_path)
        else:
            raise ValueError(f"Unsupported format for eval: {format}")
    
    def export_with_splits(
        self,
        records: list,
        output_dir: Path,
        split: Optional[DatasetSplit] = None,
        prefix: str = "train",
        format: ExportFormat = ExportFormat.JSONL,
    ) -> dict:
        """Export with train/val/test splits."""
        split = split or DatasetSplit()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Shuffle records
        shuffled = records.copy()
        random.shuffle(shuffled)
        
        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * split.train_ratio)
        val_end = train_end + int(n * split.val_ratio)
        
        # Split
        train_records = shuffled[:train_end]
        val_records = shuffled[train_end:val_end]
        test_records = shuffled[val_end:]
        
        # Export each split
        counts = {}
        ext = format.value
        
        if train_records:
            path = output_dir / f"{prefix}_train.{ext}"
            counts["train"] = self._export_jsonl(train_records, path)
        
        if val_records:
            path = output_dir / f"{prefix}_val.{ext}"
            counts["val"] = self._export_jsonl(val_records, path)
        
        if test_records:
            path = output_dir / f"{prefix}_test.{ext}"
            counts["test"] = self._export_jsonl(test_records, path)
        
        return counts
    
    def _export_jsonl(self, records: list, path: Path) -> int:
        """Export to JSONL format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            for record in records:
                if hasattr(record, 'to_dict'):
                    line = json.dumps(record.to_dict())
                else:
                    line = json.dumps(record)
                f.write(line + '\n')
        
        logger.info(f"Exported {len(records)} records to {path}")
        return len(records)
    
    def _export_parquet(self, records: list, path: Path) -> int:
        """Export to Parquet format."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Parquet export")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to flat dict for Parquet
        flat_records = []
        for record in records:
            if hasattr(record, 'to_dict'):
                flat = self._flatten_dict(record.to_dict())
            else:
                flat = self._flatten_dict(record)
            flat_records.append(flat)
        
        df = pd.DataFrame(flat_records)
        df.to_parquet(path, index=False)
        
        logger.info(f"Exported {len(records)} records to {path}")
        return len(records)
    
    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        """Flatten nested dict for Parquet."""
        flat = {}
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_dict(value, new_key))
            elif isinstance(value, list):
                flat[new_key] = json.dumps(value)
            else:
                flat[new_key] = value
        
        return flat


# =============================================================================
# DATASET BUILDER
# =============================================================================

@dataclass
class BuildStats:
    """Statistics from dataset build."""
    
    sft_total: int = 0
    sft_gold: int = 0
    dpo_total: int = 0
    eval_total: int = 0
    
    sft_splits: dict = None
    dpo_splits: dict = None
    
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.sft_splits is None:
            self.sft_splits = {}
        if self.dpo_splits is None:
            self.dpo_splits = {}
    
    def to_dict(self) -> dict:
        return {
            "sft": {
                "total": self.sft_total,
                "gold": self.sft_gold,
                "splits": self.sft_splits,
            },
            "dpo": {
                "total": self.dpo_total,
                "splits": self.dpo_splits,
            },
            "eval": {
                "total": self.eval_total,
            },
            "timing": {
                "start": self.start_time,
                "end": self.end_time,
                "duration_seconds": self.duration_seconds,
            },
        }


class DatasetBuilder:
    """Complete dataset building pipeline."""
    
    def __init__(
        self,
        split: Optional[DatasetSplit] = None,
        gold_only: bool = True,
    ):
        self.split = split or DatasetSplit()
        self.gold_only = gold_only
        self.exporter = DatasetExporter()
    
    def build(
        self,
        sft_records: list[CTv3Record],
        dpo_records: list[CTv3DPORecord],
        eval_records: list,
        output_dir: Path,
    ) -> BuildStats:
        """Build complete dataset."""
        stats = BuildStats()
        stats.start_time = datetime.utcnow().isoformat()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats.sft_total = len(sft_records)
        stats.dpo_total = len(dpo_records)
        stats.eval_total = len(eval_records)
        
        # Filter SFT by quality
        if self.gold_only:
            gold_sft = [
                r for r in sft_records
                if (hasattr(r, 'quality') and r.quality.gold) or
                   (isinstance(r, dict) and r.get('quality', {}).get('gold'))
            ]
        else:
            gold_sft = sft_records
        
        stats.sft_gold = len(gold_sft)
        
        # Export SFT
        if gold_sft:
            stats.sft_splits = self.exporter.export_with_splits(
                gold_sft,
                output_dir,
                self.split,
                prefix="sft",
            )
        
        # Export DPO
        if dpo_records:
            stats.dpo_splits = self.exporter.export_with_splits(
                dpo_records,
                output_dir,
                self.split,
                prefix="dpo",
            )
        
        # Export eval (no split - all used for evaluation)
        if eval_records:
            eval_path = output_dir / "eval_regression.jsonl"
            self.exporter._export_jsonl(eval_records, eval_path)
        
        # Write stats
        stats.end_time = datetime.utcnow().isoformat()
        
        if stats.start_time and stats.end_time:
            start = datetime.fromisoformat(stats.start_time)
            end = datetime.fromisoformat(stats.end_time)
            stats.duration_seconds = (end - start).total_seconds()
        
        with open(output_dir / "build_stats.json", 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        logger.info(f"Dataset build complete: {stats.to_dict()}")
        
        return stats
    
    def build_from_jsonl(
        self,
        sft_path: Optional[Path] = None,
        dpo_path: Optional[Path] = None,
        eval_path: Optional[Path] = None,
        output_dir: Path = Path("data/ctv3_dataset"),
    ) -> BuildStats:
        """Build dataset from JSONL files."""
        sft_records = []
        dpo_records = []
        eval_records = []
        
        # Load SFT
        if sft_path and Path(sft_path).exists():
            with open(sft_path) as f:
                for line in f:
                    if line.strip():
                        sft_records.append(CTv3Record.from_dict(json.loads(line)))
        
        # Load DPO
        if dpo_path and Path(dpo_path).exists():
            with open(dpo_path) as f:
                for line in f:
                    if line.strip():
                        dpo_records.append(CTv3DPORecord.from_dict(json.loads(line)))
        
        # Load Eval
        if eval_path and Path(eval_path).exists():
            with open(eval_path) as f:
                for line in f:
                    if line.strip():
                        eval_records.append(json.loads(line))
        
        return self.build(sft_records, dpo_records, eval_records, output_dir)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dataset Builder for CognitiveTwin V3"
    )
    parser.add_argument(
        "--sft-input",
        type=Path,
        help="Path to SFT JSONL input",
    )
    parser.add_argument(
        "--dpo-input",
        type=Path,
        help="Path to DPO JSONL input",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        help="Path to eval JSONL input",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/ctv3_dataset"),
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test split ratios (comma-separated)",
    )
    parser.add_argument(
        "--include-non-gold",
        action="store_true",
        help="Include non-gold records",
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
    
    # Parse split
    split = DatasetSplit.from_string(args.split)
    
    # Build
    builder = DatasetBuilder(
        split=split,
        gold_only=not args.include_non_gold,
    )
    
    stats = builder.build_from_jsonl(
        sft_path=args.sft_input,
        dpo_path=args.dpo_input,
        eval_path=args.eval_input,
        output_dir=args.output,
    )
    
    print("\n" + "=" * 60)
    print("Dataset Build Complete")
    print("=" * 60)
    print(f"SFT records: {stats.sft_total} (gold: {stats.sft_gold})")
    print(f"  Splits: {stats.sft_splits}")
    print(f"DPO pairs: {stats.dpo_total}")
    print(f"  Splits: {stats.dpo_splits}")
    print(f"Eval cases: {stats.eval_total}")
    print(f"Duration: {stats.duration_seconds:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()


