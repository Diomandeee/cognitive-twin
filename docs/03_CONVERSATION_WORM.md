# Phase 2B: Conversation Worm

> **Purpose**: Generate topology-consistent synthetic dialogues by walking the conversation DAG and producing controlled variants that stay consistent with trajectory coordinates, phase tags, and preferred interaction policy.
>
> **Model**: GPT 5.2 (general augmentation)
>
> **Implementation File**: `rag_plusplus/ml/cognitivetwin_v3/worms/conversation_worm.py`

---

## 1. Purpose

### 1.1. Generate Topology-Consistent Synthetic Dialogues

#### 1.1.1. Why Topology Matters
- Conversations have structure (DAG, not linear)
- Branches represent alternative paths (regenerations, edits)
- Phase determines appropriate behavior (questions allowed in opening, not in solution)
- Trajectory coordinates encode semantic-structural position

#### 1.1.2. Consistency Requirements
- New synthetic turns must preserve phase-appropriate behavior
- Trajectory coordinates must be computable for synthetic data
- Generated content must align with existing conversation flow
- Style and technical depth must match surrounding context

#### 1.1.3. Output Types
- Paraphrase variants (same intent, different wording)
- Branch completions (ideal assistant turns for friction points)
- Trajectory-preserving extensions (continue threads coherently)
- Trajectory-contrast pairs (same prompt, different phase)

### 1.2. Fix Historical Friction Points

#### 1.2.1. Friction Point Identification
- Turns where user corrected the model
- Turns classified as "unjustified clarification"
- Turns that triggered user frustration

#### 1.2.2. Repair Strategy
- Generate ideal assistant response that should have occurred
- Create continuation where correction was never needed
- Remove friction from the gold trajectory

#### 1.2.3. Training Signal
- Friction turns become DPO dispreferred examples
- Repaired turns become SFT gold examples
- The model learns: "behave like me after correction, but without needing correction"

### 1.3. Produce Phase-Aware Training Data

#### 1.3.1. Phase Definitions
| Phase ID | Name | Description | Question Policy |
|----------|------|-------------|-----------------|
| 0 | Opening | Introduction, context gathering | questions_if_required |
| 1 | Context | Deep understanding, clarification | questions_if_required |
| 2 | Solution | Active problem-solving | no_questions |
| 3 | Refinement | Iterating on solution | no_questions |
| 4 | Synthesis | Summarizing, concluding | no_questions |
| 5 | Conclusion | Final deliverables | no_questions |

#### 1.3.2. Phase-Behavior Mapping
- Opening/Context: Clarifying questions more acceptable
- Solution/Refinement: Execute immediately, no permission-seeking
- Synthesis/Conclusion: Produce deliverables, no options dumping

---

## 2. TPO Integration

### 2.1. Path Extraction

#### 2.1.1. Import TPO Pipeline

```python
from rag_plusplus.tpo.pipeline import TPOPipeline, TPOConfig
from rag_plusplus.tpo.core.path_extractor import (
    PathExtractor,
    ConversationPath,
    PathNode,
)
from rag_plusplus.tpo.core.coordinates import DLMCoordinate
from rag_plusplus.tpo.core.quality_calculator import PathQualityCalculator
```

#### 2.1.2. Configure for V3 Branch Generation

```python
@dataclass
class ConversationWormConfig:
    """Configuration for Conversation Worm."""
    
    # TPO integration
    tpo_config: TPOConfig = field(default_factory=TPOConfig)
    
    # Branch generation
    max_branches_per_friction: int = 3
    min_quality_threshold: float = 0.6
    
    # Phase behavior
    phase_question_policies: dict = field(default_factory=lambda: {
        0: "questions_if_required",  # Opening
        1: "questions_if_required",  # Context
        2: "no_questions",           # Solution
        3: "no_questions",           # Refinement
        4: "no_questions",           # Synthesis
        5: "no_questions",           # Conclusion
    })
    
    # Generation parameters
    paraphrase_count: int = 2
    extension_max_turns: int = 3
    
    # Validation
    require_no_questions_above_phase: int = 2
```

#### 2.1.3. Worm Initialization

```python
class ConversationWorm:
    """Conversation DAG traversal agent for training data generation."""
    
    def __init__(
        self,
        supabase_client,
        config: ConversationWormConfig = None,
        openai_client: OpenAI = None,
    ):
        self.client = supabase_client
        self.config = config or ConversationWormConfig()
        self.openai = openai_client or OpenAI()
        
        # Initialize TPO pipeline
        self.tpo = TPOPipeline(
            supabase_client=supabase_client,
            config=self.config.tpo_config,
        )
        
        # State
        self.processed_conversations: set[str] = set()
        self.generated_branches: list[SyntheticBranch] = []
    
    async def process_conversation(
        self,
        conversation_id: str
    ) -> list[SyntheticBranch]:
        """Process a single conversation."""
        
        # Extract paths using TPO
        results = await self.tpo.process_conversation(conversation_id)
        
        branches = []
        
        # Find friction paths
        friction_paths = self._identify_friction_paths(results.paths)
        
        for path in friction_paths:
            # Generate ideal branch
            ideal = await self._generate_ideal_branch(path)
            branches.append(ideal)
        
        # Generate paraphrases
        paraphrases = await self._generate_paraphrases(results.paths)
        branches.extend(paraphrases)
        
        # Generate extensions
        extensions = await self._generate_extensions(results.paths)
        branches.extend(extensions)
        
        self.generated_branches.extend(branches)
        self.processed_conversations.add(conversation_id)
        
        return branches
```

### 2.2. Quality Scoring

#### 2.2.1. Path Quality Computation

```python
def score_path_quality(self, path: ConversationPath) -> float:
    """Compute quality score for a path."""
    
    # Use TPO quality calculator
    calculator = PathQualityCalculator(
        weights=self.config.tpo_config.quality_weights,
    )
    
    return calculator.calculate(path)
```

#### 2.2.2. Friction Detection

```python
def _identify_friction_paths(
    self,
    paths: list[ConversationPath]
) -> list[ConversationPath]:
    """Identify paths that contain friction points."""
    
    friction_paths = []
    
    for path in paths:
        has_friction = False
        
        for node in path.nodes:
            # Check for user frustration
            if node.role == "user":
                from .corpus_surgery.quarantine import detect_frustration
                if detect_frustration(node.content):
                    has_friction = True
                    break
            
            # Check for unjustified clarification
            if node.role == "assistant":
                from .corpus_surgery.classifier import classify_assistant_turn
                
                # Get preceding user message
                user_content = self._get_preceding_user(path, node)
                
                result = classify_assistant_turn(
                    assistant_message=node.content,
                    user_message=user_content,
                    phase_id=self._get_phase(node),
                    format_constraints={},
                    directive_completeness=self._compute_completeness(user_content),
                )
                
                if result.classification.value == "unjustified":
                    has_friction = True
                    break
        
        if has_friction:
            friction_paths.append(path)
    
    return friction_paths
```

### 2.3. Coordinate Preservation

#### 2.3.1. Computing Coordinates for Synthetic Turns

```python
def compute_synthetic_coordinates(
    self,
    parent_node: PathNode,
    synthetic_content: str,
    branch_type: str
) -> DLMCoordinate:
    """Compute 5D coordinates for synthetic turn."""
    
    parent_coord = DLMCoordinate.from_dict({
        "depth": parent_node.depth,
        "sibling_order": parent_node.sibling_order,
        "homogeneity": parent_node.homogeneity,
        "temporal": parent_node.temporal,
        "complexity": parent_node.complexity,
    })
    
    # Depth increases by 1
    new_depth = parent_coord.x + 1
    
    # Sibling order depends on branch type
    if branch_type == "continuation":
        new_sibling = 0  # First response
    elif branch_type == "alternative":
        new_sibling = parent_coord.y + 1  # Additional sibling
    else:
        new_sibling = 0
    
    # Compute homogeneity (semantic similarity to parent)
    new_homogeneity = self._compute_homogeneity(
        parent_node.content,
        synthetic_content
    )
    
    # Temporal advances
    new_temporal = min(1.0, parent_coord.t + 0.05)
    
    # Complexity from content
    new_complexity = self._compute_complexity(synthetic_content)
    
    return DLMCoordinate(
        x=new_depth,
        y=new_sibling,
        z=new_homogeneity,
        t=new_temporal,
        n=new_complexity,
    )
```

#### 2.3.2. Homogeneity Computation

```python
async def _compute_homogeneity(
    self,
    parent_content: str,
    child_content: str
) -> float:
    """Compute semantic similarity between parent and child."""
    
    # Use embedding similarity
    from rag_plusplus.service.embedding import EmbedderService
    
    embedder = EmbedderService()
    
    parent_emb = await embedder.embed(parent_content)
    child_emb = await embedder.embed(child_content)
    
    # Cosine similarity
    import numpy as np
    
    similarity = np.dot(parent_emb, child_emb) / (
        np.linalg.norm(parent_emb) * np.linalg.norm(child_emb)
    )
    
    return float(similarity)
```

---

## 3. Branch Generation

### 3.1. Paraphrase Variants

#### 3.1.1. Purpose
- Robustness to how user phrases directives
- Same intent, different wording
- Teaches model to recognize directives in various forms

#### 3.1.2. Generation Strategy

```python
PARAPHRASE_SYSTEM_PROMPT = """You are a paraphrase generator for training data.

Generate {count} paraphrases of the user message that:
1. Preserve the exact same intent and meaning
2. Use different wording, sentence structure, or phrasing
3. Maintain the same level of formality
4. Keep any technical terms unchanged

Output format:
PARAPHRASE 1: ...
PARAPHRASE 2: ...
etc.
"""

async def generate_paraphrases(
    self,
    user_message: str,
    count: int = 2
) -> list[str]:
    """Generate paraphrases of user message."""
    
    response = await self.openai.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT.format(count=count)},
            {"role": "user", "content": f"Generate paraphrases for:\n\n{user_message}"}
        ],
        temperature=0.7,
    )
    
    # Parse response
    paraphrases = []
    for line in response.choices[0].message.content.split('\n'):
        if line.startswith('PARAPHRASE'):
            _, text = line.split(':', 1)
            paraphrases.append(text.strip())
    
    return paraphrases[:count]
```

#### 3.1.3. Creating Paraphrase Records

```python
@dataclass
class SyntheticBranch:
    """A synthetic branch generated from conversation."""
    
    branch_type: str  # paraphrase, ideal_response, extension, contrast
    original_conversation_id: str
    parent_node_id: str
    
    # Content
    messages: list[dict]
    
    # Coordinates
    coordinates: DLMCoordinate
    
    # Labels
    phase_id: int
    question_policy: str
    directive_completeness: float
    
    # Quality
    source: str = "convo_worm"
    is_gold: bool = False

async def _generate_paraphrases(
    self,
    paths: list[ConversationPath]
) -> list[SyntheticBranch]:
    """Generate paraphrase variants for directive prompts."""
    
    branches = []
    
    for path in paths:
        for node in path.nodes:
            if node.role != "user":
                continue
            
            # Check if directive
            completeness = self._compute_completeness(node.content)
            if completeness < 0.5:
                continue
            
            # Generate paraphrases
            paraphrases = await self.generate_paraphrases(
                node.content,
                self.config.paraphrase_count
            )
            
            for paraphrase in paraphrases:
                # Get the assistant response that followed
                assistant_response = self._get_following_assistant(path, node)
                
                if assistant_response:
                    branch = SyntheticBranch(
                        branch_type="paraphrase",
                        original_conversation_id=path.conversation_id,
                        parent_node_id=node.turn_id,
                        messages=[
                            {"role": "user", "content": paraphrase},
                            {"role": "assistant", "content": assistant_response.content},
                        ],
                        coordinates=self.compute_synthetic_coordinates(
                            node, paraphrase, "alternative"
                        ),
                        phase_id=self._get_phase(node),
                        question_policy=self._get_question_policy(node),
                        directive_completeness=completeness,
                        is_gold=True,  # Paraphrases preserve quality
                    )
                    branches.append(branch)
    
    return branches
```

### 3.2. Branch Completions (Ideal Responses)

#### 3.2.1. Purpose
- Fix historical friction points
- Generate what assistant should have said
- Remove permission-seeking from training trajectory

#### 3.2.2. Ideal Response Generation

```python
IDEAL_RESPONSE_SYSTEM_PROMPT = """You are generating the ideal assistant response for CognitiveTwin V3.

The original assistant response asked for permission or clarification when it shouldn't have.
Generate what the assistant SHOULD have said instead.

RULES:
1. Execute immediately - do not ask permission
2. If assumptions are needed, state them briefly then proceed
3. Produce the requested artifact/output
4. Do NOT end with a question
5. Match the technical level and style of the conversation

ASSUMPTION PROTOCOL:
- State assumptions as: "Assumptions: [brief list]"
- Then proceed with full response
- NO question marks in assumptions

The original conversation context is provided. Generate only the ideal assistant response.
"""

async def generate_ideal_response(
    self,
    friction_node: PathNode,
    context: list[dict],
    format_constraints: dict
) -> str:
    """Generate ideal response for friction point."""
    
    # Build context
    context_str = "\n\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in context[-4:]
    ])
    
    response = await self.openai.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": IDEAL_RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"""CONVERSATION CONTEXT:
{context_str}

USER MESSAGE (that triggered friction):
{context[-1]['content'] if context else 'N/A'}

PROBLEMATIC ASSISTANT RESPONSE (asked permission when shouldn't have):
{friction_node.content[:1000]}

FORMAT CONSTRAINTS: {format_constraints}

Generate the ideal assistant response that executes immediately:"""}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content
```

#### 3.2.3. Creating Ideal Branch Records

```python
async def _generate_ideal_branch(
    self,
    friction_path: ConversationPath
) -> SyntheticBranch:
    """Generate ideal branch for friction path."""
    
    # Find the friction node
    friction_node = None
    preceding_context = []
    
    for i, node in enumerate(friction_path.nodes):
        if node.role == "assistant":
            from .corpus_surgery.classifier import classify_assistant_turn
            
            user_content = self._get_preceding_user_content(friction_path, i)
            
            result = classify_assistant_turn(
                assistant_message=node.content,
                user_message=user_content,
                phase_id=self._get_phase(node),
                format_constraints={},
                directive_completeness=self._compute_completeness(user_content),
            )
            
            if result.classification.value == "unjustified":
                friction_node = node
                preceding_context = [
                    {"role": n.role, "content": n.content}
                    for n in friction_path.nodes[:i]
                ]
                break
    
    if not friction_node:
        return None
    
    # Generate ideal response
    ideal_content = await self.generate_ideal_response(
        friction_node,
        preceding_context,
        {}
    )
    
    # Build the repaired messages
    repaired_messages = preceding_context + [
        {"role": "assistant", "content": ideal_content}
    ]
    
    return SyntheticBranch(
        branch_type="ideal_response",
        original_conversation_id=friction_path.conversation_id,
        parent_node_id=friction_node.turn_id,
        messages=repaired_messages,
        coordinates=self.compute_synthetic_coordinates(
            friction_node, ideal_content, "alternative"
        ),
        phase_id=self._get_phase(friction_node),
        question_policy="no_questions",  # Enforced
        directive_completeness=self._compute_completeness(
            preceding_context[-1]["content"] if preceding_context else ""
        ),
        is_gold=True,
    )
```

### 3.3. Trajectory-Preserving Extensions

#### 3.3.1. Purpose
- Teach longer coherence
- Maintain trajectory through extended exchanges
- Show sustained technical analysis

#### 3.3.2. Extension Generation

```python
EXTENSION_SYSTEM_PROMPT = """You are extending a conversation for CognitiveTwin V3 training.

Generate a natural continuation of this conversation that:
1. Maintains the same technical depth and topic
2. Shows productive progression (not circular)
3. Follows the established interaction style
4. Does NOT introduce unnecessary questions from the assistant

Generate both the next user message and assistant response.

Output format:
USER: [next user message]
ASSISTANT: [assistant response that executes without asking permission]
"""

async def generate_extension(
    self,
    path: ConversationPath,
    max_turns: int = 2
) -> list[dict]:
    """Generate extension turns for a path."""
    
    # Build context from path
    context = [
        {"role": node.role, "content": node.content}
        for node in path.nodes[-6:]
    ]
    
    context_str = "\n\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}"
        for m in context
    ])
    
    extensions = []
    
    for _ in range(max_turns):
        response = await self.openai.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": EXTENSION_SYSTEM_PROMPT},
                {"role": "user", "content": f"""CONVERSATION:
{context_str}

Generate the next exchange:"""}
            ],
            temperature=0.5,
        )
        
        # Parse response
        content = response.choices[0].message.content
        
        user_match = re.search(r'USER:\s*(.+?)(?=ASSISTANT:|$)', content, re.DOTALL)
        assistant_match = re.search(r'ASSISTANT:\s*(.+)', content, re.DOTALL)
        
        if user_match and assistant_match:
            user_content = user_match.group(1).strip()
            assistant_content = assistant_match.group(1).strip()
            
            extensions.append({"role": "user", "content": user_content})
            extensions.append({"role": "assistant", "content": assistant_content})
            
            # Update context for next iteration
            context_str += f"\n\nUser: {user_content}\n\nAssistant: {assistant_content}"
        else:
            break
    
    return extensions
```

#### 3.3.3. Creating Extension Records

```python
async def _generate_extensions(
    self,
    paths: list[ConversationPath]
) -> list[SyntheticBranch]:
    """Generate extensions for high-quality paths."""
    
    branches = []
    
    # Filter to high-quality paths only
    quality_paths = [p for p in paths if (p.quality_score or 0) >= 0.7]
    
    for path in quality_paths[:10]:  # Limit extensions
        extensions = await self.generate_extension(
            path,
            self.config.extension_max_turns
        )
        
        if not extensions:
            continue
        
        # Build complete message history
        original_messages = [
            {"role": node.role, "content": node.content}
            for node in path.nodes
        ]
        
        combined = original_messages + extensions
        
        # Compute coordinates for last turn
        last_node = path.nodes[-1]
        last_extension = extensions[-1]["content"]
        
        branch = SyntheticBranch(
            branch_type="extension",
            original_conversation_id=path.conversation_id,
            parent_node_id=last_node.turn_id,
            messages=combined,
            coordinates=self.compute_synthetic_coordinates(
                last_node, last_extension, "continuation"
            ),
            phase_id=min(5, self._get_phase(last_node) + 1),  # Advance phase
            question_policy="no_questions",
            directive_completeness=0.8,  # High for extensions
            is_gold=True,
        )
        branches.append(branch)
    
    return branches
```

### 3.4. Trajectory-Contrast Pairs

#### 3.4.1. Purpose
- Teach phase-appropriate behavior
- Same prompt, different response based on phase
- Model learns context sensitivity

#### 3.4.2. Contrast Generation

```python
async def generate_contrast_pair(
    self,
    prompt: str,
    phase_a: int,
    phase_b: int
) -> tuple[str, str]:
    """Generate contrasting responses for different phases."""
    
    response_a = await self._generate_phase_response(prompt, phase_a)
    response_b = await self._generate_phase_response(prompt, phase_b)
    
    return response_a, response_b

async def _generate_phase_response(
    self,
    prompt: str,
    phase_id: int
) -> str:
    """Generate response appropriate for a specific phase."""
    
    phase_descriptions = {
        0: "Opening phase - gathering context, clarifying questions acceptable",
        1: "Context phase - deep understanding, some clarification OK",
        2: "Solution phase - actively solving, NO questions, execute immediately",
        3: "Refinement phase - iterating on solution, NO questions, just improve",
        4: "Synthesis phase - summarizing, NO questions, produce deliverables",
        5: "Conclusion phase - final output, NO questions, complete the task",
    }
    
    question_policy = self.config.phase_question_policies.get(phase_id, "no_questions")
    
    system_prompt = f"""You are responding in the {phase_descriptions.get(phase_id, 'unknown')} of a conversation.

Question policy: {question_policy}

{"You MAY ask clarifying questions if genuinely needed." if question_policy == "questions_if_required" else "Do NOT ask any questions. Execute immediately."}

Respond to the user's message appropriately for this phase.
"""
    
    response = await self.openai.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    
    return response.choices[0].message.content
```

---

## 4. Policy Enforcement

### 4.1. Question Policy by Phase

#### 4.1.1. Policy Determination

```python
def _get_question_policy(self, node: PathNode) -> str:
    """Determine question policy for a node based on phase."""
    
    phase_id = self._get_phase(node)
    return self.config.phase_question_policies.get(phase_id, "no_questions")
```

#### 4.1.2. Policy Validation

```python
def validate_question_policy(
    self,
    response: str,
    policy: str
) -> tuple[bool, list[str]]:
    """Validate response against question policy."""
    
    from .corpus_surgery.classifier import (
        compute_stall_score,
        ends_with_question,
        STRONG_PERMISSION_PHRASES,
    )
    
    errors = []
    
    if policy == "no_questions":
        # Must not ask any questions
        if ends_with_question(response):
            errors.append("Response ends with question (policy: no_questions)")
        
        stall_score = compute_stall_score(response)
        if stall_score >= 2:
            errors.append(f"Response has high stall score: {stall_score}")
        
        # Check for permission phrases
        response_lower = response.lower()
        for phrase in STRONG_PERMISSION_PHRASES:
            if phrase in response_lower:
                errors.append(f"Contains permission phrase: '{phrase}'")
                break
    
    elif policy == "questions_if_required":
        # Questions allowed but should not be gratuitous
        stall_score = compute_stall_score(response)
        if stall_score >= 5:  # Higher threshold
            errors.append(f"Too many permission-seeking phrases: {stall_score}")
    
    # questions_allowed - no restrictions
    
    return len(errors) == 0, errors
```

### 4.2. Repair Elimination

#### 4.2.1. Detecting Repair Turns

```python
def is_repair_turn(self, user_message: str, preceding_assistant: str) -> bool:
    """Detect if user is repairing/correcting the assistant."""
    
    from .corpus_surgery.quarantine import FRUSTRATION_TRIGGERS
    
    user_lower = user_message.lower()
    
    # Check for frustration triggers
    for trigger in FRUSTRATION_TRIGGERS:
        if trigger in user_lower:
            return True
    
    # Check for correction patterns
    correction_patterns = [
        r"^no[,\.]",              # "No, I meant..."
        r"^actually",             # "Actually..."
        r"that's not",            # "That's not what I asked"
        r"i already",             # "I already told you..."
        r"try again",             # "Try again"
        r"let me rephrase",       # "Let me rephrase..."
    ]
    
    for pattern in correction_patterns:
        if re.search(pattern, user_lower):
            return True
    
    return False
```

#### 4.2.2. Generating Non-Repair Trajectories

```python
async def generate_non_repair_trajectory(
    self,
    conversation: list[dict],
    repair_idx: int
) -> list[dict]:
    """Generate trajectory where repair was never needed."""
    
    # Get context up to the bad assistant turn
    context = conversation[:repair_idx - 1]  # Exclude bad turn
    
    # Get the user message before the bad turn
    user_message = conversation[repair_idx - 2]["content"]
    
    # Generate ideal response
    ideal = await self.generate_ideal_response(
        PathNode(content=conversation[repair_idx - 1]["content"], role="assistant"),
        context,
        {}
    )
    
    # Continue conversation naturally
    new_trajectory = context + [
        {"role": "assistant", "content": ideal}
    ]
    
    # Generate follow-up
    extensions = await self.generate_extension_from_messages(
        new_trajectory,
        max_turns=1
    )
    
    return new_trajectory + extensions
```

### 4.3. Format Lock

#### 4.3.1. Detecting Format Constraints

```python
def extract_format_constraints(self, user_message: str) -> dict:
    """Extract format constraints from user message."""
    
    constraints = {
        "forbid_bullets": False,
        "require_numbered": False,
        "must_return_code": False,
        "must_return_diff": False,
        "must_return_json": False,
        "must_not_omit": False,
    }
    
    user_lower = user_message.lower()
    
    # Check each constraint
    if any(p in user_lower for p in ["no bullet", "don't use bullet", "without bullet"]):
        constraints["forbid_bullets"] = True
    
    if any(p in user_lower for p in ["numbered list", "numbered steps", "number them"]):
        constraints["require_numbered"] = True
    
    if any(p in user_lower for p in ["in code", "write code", "implement", "function"]):
        constraints["must_return_code"] = True
    
    if any(p in user_lower for p in ["as json", "in json", "json format"]):
        constraints["must_return_json"] = True
    
    if any(p in user_lower for p in ["don't omit", "don't skip", "include everything", "full", "complete"]):
        constraints["must_not_omit"] = True
    
    return constraints
```

#### 4.3.2. Enforcing Format in Generation

```python
def build_format_instruction(self, constraints: dict) -> str:
    """Build format instruction for generation."""
    
    instructions = []
    
    if constraints.get("forbid_bullets"):
        instructions.append("Do NOT use bullet points. Use prose or numbered lists instead.")
    
    if constraints.get("require_numbered"):
        instructions.append("Use numbered lists for any structured content.")
    
    if constraints.get("must_return_code"):
        instructions.append("Include code in your response.")
    
    if constraints.get("must_return_json"):
        instructions.append("Return output in valid JSON format.")
    
    if constraints.get("must_not_omit"):
        instructions.append("Include ALL content - do not summarize or omit anything.")
    
    return "\n".join(instructions) if instructions else ""
```

---

## 5. Output Records

### 5.1. SFT Turn Records

```python
def create_sft_record(
    self,
    branch: SyntheticBranch
) -> dict:
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
                "depth_norm": branch.coordinates.x / 10,  # Normalize
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
            "messages": branch.messages[:-1],  # All but last
            "attachments": [],
        },
        "target": {
            "assistant_content": branch.messages[-1]["content"],
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
```

### 5.2. DPO Pair Records

```python
def create_dpo_record(
    self,
    original_messages: list[dict],
    preferred_response: str,
    dispreferred_response: str,
    branch: SyntheticBranch
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
```

---

## 6. Complete Pipeline

```python
class ConversationWormPipeline:
    """Complete Conversation Worm pipeline."""
    
    async def run(
        self,
        conversation_ids: list[str] | None = None,
        output_dir: Path = None,
    ) -> dict:
        """Run the complete pipeline."""
        
        worm = ConversationWorm(self.supabase_client)
        
        # Get conversations to process
        if conversation_ids is None:
            conversation_ids = await self._get_all_conversation_ids()
        
        sft_records = []
        dpo_records = []
        
        for conv_id in conversation_ids:
            try:
                branches = await worm.process_conversation(conv_id)
                
                for branch in branches:
                    # Create SFT record
                    sft = worm.create_sft_record(branch)
                    sft_records.append(sft)
                    
                    # Create DPO pair if it's an ideal response
                    if branch.branch_type == "ideal_response":
                        # Get original dispreferred response
                        original = self._get_original_response(conv_id, branch.parent_node_id)
                        
                        if original:
                            dpo = worm.create_dpo_record(
                                branch.messages[:-1],
                                branch.messages[-1]["content"],  # Preferred
                                original,  # Dispreferred
                                branch,
                            )
                            dpo_records.append(dpo)
            
            except Exception as e:
                logger.warning(f"Error processing {conv_id}: {e}")
                continue
        
        # Export
        if output_dir:
            self._export_jsonl(sft_records, output_dir / "convo_sft.jsonl")
            self._export_jsonl(dpo_records, output_dir / "convo_dpo.jsonl")
        
        return {
            "conversations_processed": len(conversation_ids),
            "sft_records": len(sft_records),
            "dpo_pairs": len(dpo_records),
            "branches_by_type": self._count_by_type(worm.generated_branches),
        }
```

