"""
Worms: Data augmentation agents for CognitiveTwin V3.

Worms are agents that traverse data sources to generate training data:
- RepoWorm: Traverses codebases to generate code tasks and DPO pairs
- ConversationWorm: Generates topology-consistent conversation branches
- EnhancerAgent: Canonicalizes outputs, completes content, extracts eval cases
"""

# =============================================================================
# REPO WORM (Phase 2A)
# =============================================================================

from .task_types import (
    RepoWormConfig,
    ImplementationTask,
    CompletionTask,
    RefactoringTask,
    TestTask,
    RepoTaskRecord,
    RepoAttachment,
    ParsedResponse,
)

from .code_scanner import CodeScanner
from .task_generator import TaskGenerator
from .prompt_templates import CODEX_SYSTEM_PROMPT
from .response_validator import ResponseValidator
from .dpo_generator import DPOGenerator
from .repo_worm import RepoWorm, RepoWormPipeline

# =============================================================================
# CONVERSATION WORM (Phase 2B)
# =============================================================================

from .branch_types import (
    ConversationWormConfig,
    SyntheticBranch,
    DLMCoordinate,
    PathNode,
    ConversationPath,
    BranchResult,
    ConversationWormStats,
)

from .convo_prompts import (
    PARAPHRASE_SYSTEM_PROMPT,
    IDEAL_RESPONSE_SYSTEM_PROMPT,
    EXTENSION_SYSTEM_PROMPT,
)

from .policy_enforcer import PolicyEnforcer
from .branch_generator import BranchGenerator
from .conversation_worm import ConversationWorm, ConversationWormPipeline

# =============================================================================
# ENHANCER AGENT (Phase 2C)
# =============================================================================

from .enhancer_types import (
    EnhancerConfig,
    EvalCase,
    AnnoyanceRecord,
    EnhancedRecord,
    IncompleteCodeMarker,
    PlaceholderMarker,
    UndeterminedPath,
    EnhancerStats,
)

from .canonicalizer import (
    Canonicalizer,
    PROVIDER_ISMS,
    APOLOGY_PATTERNS,
    FILLER_OPENINGS,
    PERMISSION_CLOSERS,
)

from .completer import Completer
from .annoyance_detector import AnnoyanceDetector
from .enhancer_agent import EnhancerAgent, EnhancerAgentPipeline

__all__ = [
    # ===================
    # REPO WORM (Phase 2A)
    # ===================
    # Config
    "RepoWormConfig",
    # Task types
    "ImplementationTask",
    "CompletionTask",
    "RefactoringTask",
    "TestTask",
    "RepoTaskRecord",
    "RepoAttachment",
    "ParsedResponse",
    # Components
    "CodeScanner",
    "TaskGenerator",
    "ResponseValidator",
    "DPOGenerator",
    # Main classes
    "RepoWorm",
    "RepoWormPipeline",
    # Constants
    "CODEX_SYSTEM_PROMPT",
    
    # ===========================
    # CONVERSATION WORM (Phase 2B)
    # ===========================
    # Config
    "ConversationWormConfig",
    # Types
    "SyntheticBranch",
    "DLMCoordinate",
    "PathNode",
    "ConversationPath",
    "BranchResult",
    "ConversationWormStats",
    # Components
    "PolicyEnforcer",
    "BranchGenerator",
    # Main classes
    "ConversationWorm",
    "ConversationWormPipeline",
    # Constants
    "PARAPHRASE_SYSTEM_PROMPT",
    "IDEAL_RESPONSE_SYSTEM_PROMPT",
    "EXTENSION_SYSTEM_PROMPT",
    
    # ========================
    # ENHANCER AGENT (Phase 2C)
    # ========================
    # Config
    "EnhancerConfig",
    # Types
    "EvalCase",
    "AnnoyanceRecord",
    "EnhancedRecord",
    "IncompleteCodeMarker",
    "PlaceholderMarker",
    "UndeterminedPath",
    "EnhancerStats",
    # Components
    "Canonicalizer",
    "Completer",
    "AnnoyanceDetector",
    # Main classes
    "EnhancerAgent",
    "EnhancerAgentPipeline",
    # Constants
    "PROVIDER_ISMS",
    "APOLOGY_PATTERNS",
    "FILLER_OPENINGS",
    "PERMISSION_CLOSERS",
]

