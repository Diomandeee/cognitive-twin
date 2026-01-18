#!/usr/bin/env python3
"""Training script for CognitiveTwin.

This script trains a CognitiveTwin model on conversation data from Supabase.

Usage:
    python scripts/train_cognitive_twin.py --preset balanced --epochs 10
    python scripts/train_cognitive_twin.py --config config/cognitive_twin.yaml
    python scripts/train_cognitive_twin.py --resume checkpoints/checkpoint_epoch_5.pt

Environment Variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_ANON_KEY: Supabase anon key
    WANDB_API_KEY: (optional) Weights & Biases API key for logging
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cognitive_twin.framework import (
    CognitiveTwin,
    CognitiveTwinConfig,
    CognitiveTwinTrainer,
    create_cognitive_twin,
    create_trainer_from_config,
    load_training_data_from_supabase,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"training_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CognitiveTwin on Supabase conversation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "accurate"],
        help="Configuration preset (default: balanced)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file (overrides preset)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )

    # Data parameters
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of turns to load (for testing)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=32,
        help="Maximum context length (default: 32)",
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints/cognitive_twin"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)",
    )

    # Hardware parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cognitive-twin",
        help="W&B project name",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, small data)",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> CognitiveTwinConfig:
    """Load configuration from preset or file.

    Args:
        args: Command line arguments.

    Returns:
        CognitiveTwinConfig instance.
    """
    if args.config:
        # Load from YAML file
        import yaml

        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = CognitiveTwinConfig(**config_dict)
    else:
        # Use preset
        if args.preset == "fast":
            config = CognitiveTwinConfig.fast()
        elif args.preset == "accurate":
            config = CognitiveTwinConfig.accurate()
        else:
            config = CognitiveTwinConfig.balanced()

    # Override with command line arguments
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.warmup_ratio = args.warmup_ratio
    config.device = args.device
    config.num_workers = args.num_workers
    config.log_every = args.log_every
    config.save_every = args.save_every
    config.reasoning_encoder.max_context_length = args.max_context

    return config


async def load_data(args: argparse.Namespace) -> tuple:
    """Load training data from Supabase.

    Args:
        args: Command line arguments.

    Returns:
        Tuple of (train_data, val_data).
    """
    # Get Supabase credentials
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_ANON_KEY environment variables required"
        )

    logger.info("Loading data from Supabase...")

    # Load data
    limit = args.limit or (100 if args.debug else None)
    train_data, val_data = await load_training_data_from_supabase(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        limit=limit,
    )

    logger.info(f"Loaded {len(train_data)} training turns, {len(val_data)} validation turns")

    return train_data, val_data


def setup_wandb(args: argparse.Namespace, config: CognitiveTwinConfig) -> Optional[Any]:
    """Setup Weights & Biases logging.

    Args:
        args: Command line arguments.
        config: Training configuration.

    Returns:
        W&B run object or None.
    """
    if not args.wandb:
        return None

    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config={
                "preset": args.preset,
                "epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "embed_dim": config.reasoning_encoder.embed_dim,
                "style_dim": config.style_projector.style_dim,
                "num_patterns": config.reasoning_encoder.num_pattern_types,
            },
            name=f"cognitive-twin-{datetime.now():%Y%m%d_%H%M%S}",
        )
        logger.info(f"W&B run initialized: {run.url}")
        return run
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
        return None


def wandb_callback(trainer: CognitiveTwinTrainer, epoch: int, metrics: Any) -> None:
    """W&B logging callback.

    Args:
        trainer: The trainer instance.
        epoch: Current epoch.
        metrics: Training metrics.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics.to_dict(), step=epoch)
    except ImportError:
        pass


async def main() -> int:
    """Main training function.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    logger.info("=" * 60)
    logger.info("CognitiveTwin Training")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args)
    logger.info(f"Configuration: preset={args.preset}")
    logger.info(f"  - epochs: {config.num_epochs}")
    logger.info(f"  - batch_size: {config.batch_size}")
    logger.info(f"  - learning_rate: {config.learning_rate}")
    logger.info(f"  - device: {config.device}")

    # Setup W&B
    wandb_run = setup_wandb(args, config)

    # Load data
    train_data, val_data = await load_data(args)

    # Create model
    logger.info("Creating CognitiveTwin model...")
    model = create_cognitive_twin(preset=args.preset)

    # Create trainer
    callbacks = [wandb_callback] if wandb_run else []
    trainer = create_trainer_from_config(
        config=config,
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.callbacks = callbacks

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    results = trainer.train()

    # Save final model
    model_path = args.checkpoint_dir / "final_model"
    trainer.model.save(model_path)
    logger.info(f"Final model saved to: {model_path}")

    # Log final results
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best epoch: {results['best_epoch'] + 1}")
    logger.info(f"Total patterns learned: {results['total_patterns_learned']}")
    logger.info(f"Conversations processed: {results['total_conversations']}")

    # Finish W&B
    if wandb_run:
        import wandb

        wandb.finish()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)
