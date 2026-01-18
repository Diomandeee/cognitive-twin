"""
Branch Generator for Conversation Worm.

Generates synthetic branches:
- Paraphrases
- Ideal responses
- Extensions
- Contrast pairs
"""

import re
import logging
from typing import Optional

from .branch_types import (
    ConversationWormConfig,
    SyntheticBranch,
    DLMCoordinate,
    PathNode,
    ConversationPath,
)
from .convo_prompts import (
    PARAPHRASE_SYSTEM_PROMPT,
    IDEAL_RESPONSE_SYSTEM_PROMPT,
    EXTENSION_SYSTEM_PROMPT,
    format_paraphrase_prompt,
    format_ideal_response_prompt,
    format_extension_prompt,
    format_contrast_prompt,
    get_phase_system_prompt,
)
from .policy_enforcer import PolicyEnforcer


logger = logging.getLogger(__name__)


class BranchGenerator:
    """Generates synthetic branches from conversations."""
    
    def __init__(
        self,
        openai_client,
        config: Optional[ConversationWormConfig] = None,
        v2_generator=None,
    ):
        self.openai = openai_client
        self.config = config or ConversationWormConfig()
        self.enforcer = PolicyEnforcer(self.config)
        self.v2_generator = v2_generator
        self.use_v2 = v2_generator is not None
    
    # =========================================================================
    # PARAPHRASES
    # =========================================================================
    
    async def generate_paraphrases(
        self,
        user_message: str,
        count: int = 2,
    ) -> list[str]:
        """Generate paraphrases of user message."""
        system_prompt = PARAPHRASE_SYSTEM_PROMPT.format(count=count)
        user_prompt = format_paraphrase_prompt(user_message, count)
        
        try:
            response = await self.openai.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature_paraphrase,
            )
            
            # Parse response
            paraphrases = []
            content = response.get("content", "")
            
            for line in content.split('\n'):
                line = line.strip()
                if line.upper().startswith('PARAPHRASE'):
                    match = re.match(r'PARAPHRASE\s*\d*\s*:\s*(.+)', line, re.IGNORECASE)
                    if match:
                        paraphrases.append(match.group(1).strip())
            
            return paraphrases[:count]
        
        except Exception as e:
            logger.warning(f"Failed to generate paraphrases: {e}")
            return []
    
    async def create_paraphrase_branches(
        self,
        path: ConversationPath,
    ) -> list[SyntheticBranch]:
        """Generate paraphrase branches for directive prompts in a path."""
        branches = []
        
        for i, node in enumerate(path.nodes):
            if node.role != "user":
                continue
            
            # Check if directive
            completeness = self.enforcer.compute_directive_completeness(node.content)
            if completeness < 0.5:
                continue
            
            # Generate paraphrases
            paraphrases = await self.generate_paraphrases(
                node.content,
                self.config.paraphrase_count,
            )
            
            # Get the assistant response that followed
            assistant_response = self._get_following_assistant(path.nodes, i)
            
            if not assistant_response:
                continue
            
            for paraphrase in paraphrases:
                # Compute coordinates
                coordinates = self._compute_branch_coordinates(
                    node, paraphrase, "alternative"
                )
                
                phase_id = self.enforcer.estimate_phase(
                    path.to_messages(), i
                )
                
                branch = SyntheticBranch(
                    branch_type="paraphrase",
                    original_conversation_id=path.conversation_id,
                    parent_node_id=node.turn_id,
                    messages=[
                        {"role": "user", "content": paraphrase},
                        {"role": "assistant", "content": assistant_response.content},
                    ],
                    coordinates=coordinates,
                    phase_id=phase_id,
                    question_policy=self.enforcer.get_question_policy(phase_id),
                    directive_completeness=completeness,
                    is_gold=True,  # Paraphrases preserve quality
                )
                branches.append(branch)
        
        return branches
    
    def _get_following_assistant(
        self,
        nodes: list[PathNode],
        user_idx: int,
    ) -> Optional[PathNode]:
        """Get the assistant node following a user node."""
        for j in range(user_idx + 1, len(nodes)):
            if nodes[j].role == "assistant":
                return nodes[j]
        return None
    
    # =========================================================================
    # IDEAL RESPONSES
    # =========================================================================
    
    async def generate_ideal_response(
        self,
        friction_content: str,
        context: list[dict],
        format_constraints: dict,
    ) -> str:
        """Generate ideal response for friction point.
        
        Uses V2 generator when available (preferred for style-aligned outputs),
        otherwise falls back to OpenAI.
        """
        # Try V2 generator first (preferred - has learned user's style)
        if self.use_v2 and self.v2_generator is not None:
            try:
                # Extract user message from context
                user_message = ""
                for turn in reversed(context):
                    if turn.get("role") == "user":
                        user_message = turn.get("content", "")
                        break
                
                if not user_message:
                    user_message = friction_content
                
                result = await self.v2_generator.generate_ideal_response(
                    user_message=user_message,
                    context="\n".join([f"{t['role']}: {t['content'][:200]}" for t in context[-4:]]),
                )
                
                if result.success:
                    return result.content
                
            except Exception as e:
                logger.warning(f"V2 generation failed, falling back to OpenAI: {e}")
        
        # Fallback to OpenAI
        user_prompt = format_ideal_response_prompt(
            context, friction_content, format_constraints
        )
        
        try:
            response = await self.openai.chat_complete(
                messages=[
                    {"role": "system", "content": IDEAL_RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature_ideal,
            )
            
            return response.get("content", "")
        
        except Exception as e:
            logger.warning(f"Failed to generate ideal response: {e}")
            return ""
    
    async def create_ideal_branch(
        self,
        friction_path: ConversationPath,
    ) -> Optional[SyntheticBranch]:
        """Generate ideal branch for friction path."""
        # Find the friction node
        friction_node = None
        friction_idx = None
        preceding_context = []
        
        for i, node in enumerate(friction_path.nodes):
            if node.role == "assistant":
                # Check for unjustified clarification
                user_content = self._get_preceding_user_content(friction_path.nodes, i)
                
                if self._is_unjustified_clarification(node.content, user_content):
                    friction_node = node
                    friction_idx = i
                    preceding_context = [
                        {"role": n.role, "content": n.content}
                        for n in friction_path.nodes[:i]
                    ]
                    break
        
        if not friction_node:
            return None
        
        # Get format constraints from user message
        user_message = preceding_context[-1]["content"] if preceding_context else ""
        format_constraints = self.enforcer.extract_format_constraints(user_message)
        
        # Generate ideal response
        ideal_content = await self.generate_ideal_response(
            friction_node.content,
            preceding_context,
            format_constraints,
        )
        
        if not ideal_content:
            return None
        
        # Validate the generated response
        is_valid, errors = self.enforcer.validate_question_policy(
            ideal_content, "no_questions"
        )
        
        if not is_valid:
            logger.warning(f"Generated ideal response failed validation: {errors}")
            # Could retry or return None
        
        # Build the repaired messages
        repaired_messages = preceding_context + [
            {"role": "assistant", "content": ideal_content}
        ]
        
        # Compute coordinates
        coordinates = self._compute_branch_coordinates(
            friction_node, ideal_content, "alternative"
        )
        
        phase_id = self.enforcer.estimate_phase(
            friction_path.to_messages(), friction_idx or 0
        )
        
        return SyntheticBranch(
            branch_type="ideal_response",
            original_conversation_id=friction_path.conversation_id,
            parent_node_id=friction_node.turn_id,
            messages=repaired_messages,
            coordinates=coordinates,
            phase_id=phase_id,
            question_policy="no_questions",  # Enforced
            directive_completeness=self.enforcer.compute_directive_completeness(
                user_message
            ),
            is_gold=True,
        )
    
    def _get_preceding_user_content(
        self,
        nodes: list[PathNode],
        assistant_idx: int,
    ) -> str:
        """Get the user content before an assistant node."""
        for j in range(assistant_idx - 1, -1, -1):
            if nodes[j].role == "user":
                return nodes[j].content
        return ""
    
    def _is_unjustified_clarification(
        self,
        assistant_content: str,
        user_content: str,
    ) -> bool:
        """Check if assistant response is unjustified clarification."""
        try:
            from ..corpus_surgery.classifier import classify_assistant_turn
            
            result = classify_assistant_turn(
                assistant_message=assistant_content,
                user_message=user_content,
                phase_id=2,  # Assume solution phase
                format_constraints={},
                directive_completeness=self.enforcer.compute_directive_completeness(
                    user_content
                ),
            )
            
            return result.classification.value == "unjustified"
        
        except ImportError:
            # Fallback: use stall score
            stall_score = self.enforcer._compute_stall_score(assistant_content)
            completeness = self.enforcer.compute_directive_completeness(user_content)
            
            return stall_score >= 3 and completeness >= 0.5
    
    # =========================================================================
    # EXTENSIONS
    # =========================================================================
    
    async def generate_extension(
        self,
        context: list[dict],
        max_turns: int = 2,
    ) -> list[dict]:
        """Generate extension turns for a conversation."""
        extensions = []
        current_context = context.copy()
        
        for _ in range(max_turns):
            user_prompt = format_extension_prompt(current_context)
            
            try:
                response = await self.openai.chat_complete(
                    messages=[
                        {"role": "system", "content": EXTENSION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature_extension,
                )
                
                content = response.get("content", "")
                
                # Parse response
                user_match = re.search(
                    r'USER:\s*(.+?)(?=ASSISTANT:|$)', content, re.DOTALL | re.IGNORECASE
                )
                assistant_match = re.search(
                    r'ASSISTANT:\s*(.+)', content, re.DOTALL | re.IGNORECASE
                )
                
                if user_match and assistant_match:
                    user_content = user_match.group(1).strip()
                    assistant_content = assistant_match.group(1).strip()
                    
                    # Validate assistant response
                    is_valid, errors = self.enforcer.validate_question_policy(
                        assistant_content, "no_questions"
                    )
                    
                    if not is_valid:
                        logger.debug(f"Extension failed validation: {errors}")
                        # Continue anyway for now
                    
                    extensions.append({"role": "user", "content": user_content})
                    extensions.append({"role": "assistant", "content": assistant_content})
                    
                    # Update context for next iteration
                    current_context.append({"role": "user", "content": user_content})
                    current_context.append({"role": "assistant", "content": assistant_content})
                else:
                    break
            
            except Exception as e:
                logger.warning(f"Failed to generate extension: {e}")
                break
        
        return extensions
    
    async def create_extension_branches(
        self,
        paths: list[ConversationPath],
        min_quality: float = 0.7,
    ) -> list[SyntheticBranch]:
        """Generate extensions for high-quality paths."""
        branches = []
        
        # Filter to high-quality paths only
        quality_paths = [
            p for p in paths
            if (p.quality_score or 0) >= min_quality
        ]
        
        for path in quality_paths[:10]:  # Limit extensions
            context = path.to_messages()
            
            extensions = await self.generate_extension(
                context,
                self.config.extension_max_turns,
            )
            
            if not extensions:
                continue
            
            # Build complete message history
            combined = context + extensions
            
            # Compute coordinates for last turn
            last_node = path.nodes[-1] if path.nodes else PathNode()
            last_extension = extensions[-1]["content"] if extensions else ""
            
            coordinates = self._compute_branch_coordinates(
                last_node, last_extension, "continuation"
            )
            
            phase_id = min(5, self.enforcer.estimate_phase(
                combined, len(combined) - 1
            ) + 1)  # Advance phase
            
            branch = SyntheticBranch(
                branch_type="extension",
                original_conversation_id=path.conversation_id,
                parent_node_id=last_node.turn_id,
                messages=combined,
                coordinates=coordinates,
                phase_id=phase_id,
                question_policy="no_questions",
                directive_completeness=0.8,  # High for extensions
                is_gold=True,
            )
            branches.append(branch)
        
        return branches
    
    # =========================================================================
    # CONTRAST PAIRS
    # =========================================================================
    
    async def generate_contrast_pair(
        self,
        prompt: str,
        phase_a: int,
        phase_b: int,
    ) -> tuple[Optional[str], Optional[str]]:
        """Generate contrasting responses for different phases."""
        policy_a = self.enforcer.get_question_policy(phase_a)
        policy_b = self.enforcer.get_question_policy(phase_b)
        
        user_prompt = format_contrast_prompt(
            prompt, phase_a, phase_b, policy_a, policy_b
        )
        
        try:
            response = await self.openai.chat_complete(
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
            )
            
            content = response.get("content", "")
            
            # Parse responses
            response_a_match = re.search(
                rf'RESPONSE A.*?:\s*(.+?)(?=RESPONSE B|$)',
                content, re.DOTALL | re.IGNORECASE
            )
            response_b_match = re.search(
                rf'RESPONSE B.*?:\s*(.+?)$',
                content, re.DOTALL | re.IGNORECASE
            )
            
            response_a = response_a_match.group(1).strip() if response_a_match else None
            response_b = response_b_match.group(1).strip() if response_b_match else None
            
            return response_a, response_b
        
        except Exception as e:
            logger.warning(f"Failed to generate contrast pair: {e}")
            return None, None
    
    async def generate_phase_response(
        self,
        prompt: str,
        phase_id: int,
    ) -> Optional[str]:
        """Generate response appropriate for a specific phase."""
        question_policy = self.enforcer.get_question_policy(phase_id)
        system_prompt = get_phase_system_prompt(phase_id, question_policy)
        
        try:
            response = await self.openai.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            
            return response.get("content", "")
        
        except Exception as e:
            logger.warning(f"Failed to generate phase response: {e}")
            return None
    
    # =========================================================================
    # COORDINATE COMPUTATION
    # =========================================================================
    
    def _compute_branch_coordinates(
        self,
        parent_node: PathNode,
        new_content: str,
        branch_type: str,
    ) -> DLMCoordinate:
        """Compute 5D coordinates for synthetic turn."""
        # Depth increases by 1
        new_depth = parent_node.depth + 1
        
        # Sibling order depends on branch type
        if branch_type == "continuation":
            new_sibling = 0  # First response
        elif branch_type == "alternative":
            new_sibling = parent_node.sibling_order + 1  # Additional sibling
        else:
            new_sibling = 0
        
        # Homogeneity placeholder (would use embeddings)
        new_homogeneity = 0.8 if branch_type == "continuation" else 0.6
        
        # Temporal advances
        new_temporal = min(1.0, parent_node.temporal + 0.05)
        
        # Complexity from content length and code presence
        new_complexity = self._compute_complexity(new_content)
        
        return DLMCoordinate(
            x=float(new_depth),
            y=float(new_sibling),
            z=new_homogeneity,
            t=new_temporal,
            n=new_complexity,
        )
    
    def _compute_complexity(self, content: str) -> float:
        """Compute content complexity score."""
        score = 0.0
        
        # Length factor
        length = len(content)
        if length > 2000:
            score += 0.3
        elif length > 1000:
            score += 0.2
        elif length > 500:
            score += 0.1
        
        # Code presence
        if '```' in content:
            score += 0.3
        
        # Technical terms
        technical_terms = [
            'function', 'class', 'import', 'async', 'await',
            'algorithm', 'complexity', 'implementation', 'architecture',
        ]
        
        content_lower = content.lower()
        term_count = sum(1 for term in technical_terms if term in content_lower)
        score += min(0.4, term_count * 0.05)
        
        return min(1.0, score)

