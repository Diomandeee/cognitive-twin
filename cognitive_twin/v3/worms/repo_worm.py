"""
Repo Worm: Codebase traversal agent for training data generation.

Main orchestrator that:
1. Scans repository for tasks
2. Generates preferred responses using GPT 5.2 Codex
3. Generates dispreferred responses (stalling)
4. Creates validated SFT records and DPO pairs
"""

import json
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime

from .task_types import (
    RepoWormConfig,
    Task,
    ImplementationTask,
    CompletionTask,
    RefactoringTask,
    TestTask,
    RepoTaskRecord,
    RepoAttachment,
    ParsedResponse,
    TaskResult,
)
from .code_scanner import CodeScanner
from .task_generator import TaskGenerator
from .prompt_templates import (
    CODEX_SYSTEM_PROMPT,
    format_prompt,
    prepare_context_window,
)
from .response_validator import ResponseValidator
from .dpo_generator import DPOGenerator
from ..api.openai_client import V3OpenAIClient, ClientConfig
from ..corpus_surgery.types import DPOPair


@dataclass
class RepoWormStats:
    """Statistics from Repo Worm execution."""
    
    total_tasks: int = 0
    implementation_tasks: int = 0
    completion_tasks: int = 0
    refactoring_tasks: int = 0
    test_tasks: int = 0
    
    processed_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    
    sft_records: int = 0
    dpo_pairs: int = 0
    
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "tasks": {
                "total": self.total_tasks,
                "implementation": self.implementation_tasks,
                "completion": self.completion_tasks,
                "refactoring": self.refactoring_tasks,
                "test": self.test_tasks,
            },
            "processing": {
                "processed": self.processed_tasks,
                "successful": self.successful_tasks,
                "failed": self.failed_tasks,
            },
            "outputs": {
                "sft_records": self.sft_records,
                "dpo_pairs": self.dpo_pairs,
            },
            "timing": {
                "start": self.start_time,
                "end": self.end_time,
                "duration_seconds": self.duration_seconds,
            },
        }


class RepoWorm:
    """Codebase traversal agent for training data generation."""
    
    def __init__(
        self,
        repo_path: Path,
        config: Optional[RepoWormConfig] = None,
        openai_client: Optional[V3OpenAIClient] = None,
    ):
        self.repo_path = Path(repo_path)
        self.config = config or RepoWormConfig()
        
        # Initialize components
        self.scanner = CodeScanner(repo_path, self.config)
        self.task_generator = TaskGenerator(repo_path, self.config, self.scanner)
        self.validator = ResponseValidator()
        self.dpo_generator = DPOGenerator()
        
        # Initialize OpenAI client
        if openai_client:
            self.client = openai_client
        else:
            openai_config = ClientConfig(
                default_model=self.config.codex_model,
                default_temperature=self.config.codex_temperature,
                default_max_tokens=self.config.codex_max_tokens,
            )
            self.client = V3OpenAIClient(openai_config)
        
        # State
        self.tasks: list[Task] = []
        self.results: list[TaskResult] = []
        self.sft_records: list[RepoTaskRecord] = []
        self.dpo_pairs: list[DPOPair] = []
        self.stats = RepoWormStats()
    
    # =========================================================================
    # TASK GENERATION
    # =========================================================================
    
    def generate_all_tasks(self) -> list[Task]:
        """Generate all types of tasks from the repository."""
        self.tasks = self.task_generator.generate_all_tasks()
        
        # Update stats
        self.stats.total_tasks = len(self.tasks)
        self.stats.implementation_tasks = sum(
            1 for t in self.tasks if isinstance(t, ImplementationTask)
        )
        self.stats.completion_tasks = sum(
            1 for t in self.tasks if isinstance(t, CompletionTask)
        )
        self.stats.refactoring_tasks = sum(
            1 for t in self.tasks if isinstance(t, RefactoringTask)
        )
        self.stats.test_tasks = sum(
            1 for t in self.tasks if isinstance(t, TestTask)
        )
        
        return self.tasks
    
    # =========================================================================
    # RESPONSE GENERATION
    # =========================================================================
    
    async def call_codex(
        self,
        prompt: str,
        context: str,
        task_type: str,
    ) -> str:
        """Call GPT 5.2 Codex with low temperature for deterministic output."""
        full_prompt = f"{context}\n\n{prompt}"
        
        messages = [
            {"role": "system", "content": CODEX_SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ]
        
        response = await self.client.chat_complete_async(
            messages=messages,
            temperature=self.config.codex_temperature,
            max_tokens=self.config.codex_max_tokens,
            model=self.config.codex_model,
        )
        
        return response
    
    async def generate_preferred_response(self, task: Task) -> str:
        """Generate preferred response that executes immediately."""
        prompt = format_prompt(task)
        context = prepare_context_window(task)
        
        response = await self.call_codex(prompt, context, task.task_type)
        
        return response
    
    # =========================================================================
    # TASK PROCESSING
    # =========================================================================
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process a single task: generate, validate, create records."""
        result = TaskResult(task=task, response=ParsedResponse())
        
        try:
            # Generate preferred response
            raw_response = await self.generate_preferred_response(task)
            
            # Parse response
            parsed = self.validator.parse_codex_response(raw_response)
            
            # Get reference code for pattern matching
            reference_code = ""
            if isinstance(task, ImplementationTask):
                if task.related_implementations:
                    reference_code = task.related_implementations[0]
            
            # Get existing imports
            existing_imports = set()
            file_path = getattr(task, 'target_file', None) or getattr(task, 'file', None)
            if file_path:
                imports = self.scanner.get_import_context(Path(file_path))
                for line in imports.split('\n'):
                    if line.strip().startswith('import '):
                        existing_imports.add(line.split()[1].split('.')[0])
                    elif line.strip().startswith('from '):
                        parts = line.split()
                        if len(parts) >= 2:
                            existing_imports.add(parts[1].split('.')[0])
            
            # Validate response
            validated = self.validator.validate_response(
                parsed, task, reference_code, existing_imports
            )
            
            result.response = validated
            
            if validated.is_valid:
                # Create SFT record
                sft_record = self._create_sft_record(task, validated)
                result.sft_record = sft_record
                self.sft_records.append(sft_record)
                
                result.success = True
                self.stats.successful_tasks += 1
            else:
                result.error = "; ".join(validated.errors)
                self.stats.failed_tasks += 1
            
            # Generate DPO pairs (even for failed preferred responses)
            if self.config.generate_dispreferred:
                preferred_text = validated.code or raw_response
                dpo_pairs = self.dpo_generator.create_all_dpo_pairs(
                    task, preferred_text
                )
                result.dpo_pairs = dpo_pairs
                self.dpo_pairs.extend(dpo_pairs)
            
        except Exception as e:
            result.error = str(e)
            result.success = False
            self.stats.failed_tasks += 1
        
        self.stats.processed_tasks += 1
        return result
    
    async def process_batch(
        self,
        tasks: list[Task],
        concurrency: int = None,
    ) -> list[TaskResult]:
        """Process multiple tasks with concurrency control."""
        concurrency = concurrency or self.config.max_concurrent_tasks
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(task: Task) -> TaskResult:
            async with semaphore:
                return await self.process_task(task)
        
        results = await asyncio.gather(
            *[process_with_semaphore(t) for t in tasks],
            return_exceptions=True
        )
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task=tasks[i],
                    response=ParsedResponse(),
                    error=str(result),
                    success=False,
                ))
            else:
                processed_results.append(result)
        
        self.results.extend(processed_results)
        return processed_results
    
    # =========================================================================
    # RECORD CREATION
    # =========================================================================
    
    def _create_sft_record(
        self,
        task: Task,
        response: ParsedResponse,
    ) -> RepoTaskRecord:
        """Create an SFT training record from a task and response."""
        # Build prompt
        prompt = format_prompt(task)
        
        # Create attachment for repo context
        file_path = getattr(task, 'target_file', None) or getattr(task, 'file', None)
        attachments = []
        if file_path:
            attachment = RepoAttachment(
                repo=str(self.repo_path),
                path=file_path,
                span={
                    "start_line": getattr(task, 'target_line', 0) or getattr(task, 'line', 0),
                    "end_line": getattr(task, 'target_line', 0) or getattr(task, 'line', 0) + 50,
                },
            )
            attachments.append(attachment.to_dict())
        
        # Determine task tags
        task_tags = {
            "task_type": task.task_type,
            "prompt_class": "directive",
            "repo_task": {
                "module": str(file_path) if file_path else "",
                "symbols": [],
                "build_required": True,
                "tests_required": isinstance(task, TestTask),
            }
        }
        
        # Quality labeling
        quality = self.dpo_generator.label_quality(
            task,
            response.code or response.raw_response,
            response.is_valid,
            response.errors,
        )
        
        record = RepoTaskRecord(
            record_type="repo_task",
            source={
                "origin": "repo_worm",
                "provider": self.config.codex_model,
                "source_id": task.task_id if hasattr(task, 'task_id') else "",
                "created_at_utc": datetime.utcnow().isoformat(),
            },
            input={
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "attachments": attachments,
            },
            target={
                "assistant_content": response.code or response.raw_response,
                "structured": {
                    "diff_unified": response.diff or "",
                    "json": {},
                }
            },
            tags=task_tags,
            quality=quality,
        )
        
        return record
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_sft(self, output_path: Path):
        """Export SFT records to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for record in self.sft_records:
                f.write(json.dumps(record.to_dict()) + '\n')
        
        self.stats.sft_records = len(self.sft_records)
        print(f"Exported {len(self.sft_records)} SFT records to {output_path}")
    
    def export_dpo(self, output_path: Path):
        """Export DPO pairs to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pair in self.dpo_pairs:
                f.write(json.dumps(pair.to_training_format()) + '\n')
        
        self.stats.dpo_pairs = len(self.dpo_pairs)
        print(f"Exported {len(self.dpo_pairs)} DPO pairs to {output_path}")
    
    def export_stats(self, output_path: Path):
        """Export stats to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        print(f"Exported stats to {output_path}")


class RepoWormPipeline:
    """Complete Repo Worm pipeline for training data generation."""
    
    def __init__(
        self,
        config: Optional[RepoWormConfig] = None,
        openai_client: Optional[V3OpenAIClient] = None,
    ):
        self.config = config or RepoWormConfig()
        self.openai_client = openai_client
    
    async def run(
        self,
        repo_path: Path,
        output_dir: Path,
    ) -> RepoWormStats:
        """Run the complete pipeline."""
        repo_path = Path(repo_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize worm
        worm = RepoWorm(repo_path, self.config, self.openai_client)
        worm.stats.start_time = datetime.now().isoformat()
        start = datetime.now()
        
        print(f"Scanning repository: {repo_path}")
        
        # Generate tasks
        tasks = worm.generate_all_tasks()
        print(f"Found {len(tasks)} tasks:")
        print(f"  - Implementation: {worm.stats.implementation_tasks}")
        print(f"  - Completion: {worm.stats.completion_tasks}")
        print(f"  - Refactoring: {worm.stats.refactoring_tasks}")
        print(f"  - Test: {worm.stats.test_tasks}")
        
        if not tasks:
            print("No tasks found.")
            return worm.stats
        
        # Process tasks
        print(f"\nProcessing {len(tasks)} tasks...")
        await worm.process_batch(tasks)
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        worm.export_sft(output_dir / f"repo_sft_{timestamp}.jsonl")
        worm.export_dpo(output_dir / f"repo_dpo_{timestamp}.jsonl")
        worm.export_stats(output_dir / f"repo_stats_{timestamp}.json")
        
        # Finalize stats
        worm.stats.end_time = datetime.now().isoformat()
        worm.stats.duration_seconds = (datetime.now() - start).total_seconds()
        
        print(f"\nPipeline complete in {worm.stats.duration_seconds:.2f}s")
        print(f"  Successful: {worm.stats.successful_tasks}")
        print(f"  Failed: {worm.stats.failed_tasks}")
        print(f"  SFT records: {worm.stats.sft_records}")
        print(f"  DPO pairs: {worm.stats.dpo_pairs}")
        
        return worm.stats


def run_pipeline_sync(
    repo_path: Path,
    output_dir: Path,
    config: Optional[RepoWormConfig] = None,
) -> RepoWormStats:
    """Synchronous wrapper for pipeline execution."""
    pipeline = RepoWormPipeline(config)
    return asyncio.run(pipeline.run(repo_path, output_dir))


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Repo Worm Pipeline")
    parser.add_argument("--repo", "-r", required=True, help="Repository path")
    parser.add_argument("--output", "-o", default="data/repo_worm_output", help="Output directory")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Max concurrent tasks")
    
    args = parser.parse_args()
    
    config = RepoWormConfig(max_concurrent_tasks=args.concurrency)
    
    stats = run_pipeline_sync(
        Path(args.repo),
        Path(args.output),
        config,
    )
    
    print("\nFinal stats:")
    print(json.dumps(stats.to_dict(), indent=2))

