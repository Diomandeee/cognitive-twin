"""
Evaluation Pipeline for CognitiveTwin V3.

Complete orchestration of evaluation including:
- Test suite execution
- Result collection
- Report generation
- CLI interface

Usage:
    # As a module
    python -m rag_plusplus.ml.cognitivetwin_v3.eval.pipeline \
        --model-id your-model-id \
        --output-dir eval_results/

    # Critical tests only
    python -m rag_plusplus.ml.cognitivetwin_v3.eval.pipeline \
        --model-id your-model-id \
        --priority critical
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

from .types import (
    EvalConfig,
    EvalSummary,
    TestResult,
    TestPriority,
    TestCategory,
)
from .suite import RegressionTestSuite
from .runner import RegressionTestRunner
from .reporter import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Complete evaluation pipeline.
    
    Orchestrates test suite execution, result collection,
    and report generation.
    
    Usage:
        config = EvalConfig(model_id="your-model-id")
        pipeline = EvaluationPipeline(config)
        
        # Run full evaluation
        summary = await pipeline.run()
        
        # Run critical tests only
        summary = await pipeline.run(priority=TestPriority.CRITICAL)
    """
    
    def __init__(self, config: EvalConfig):
        """
        Initialize evaluation pipeline.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.suite = RegressionTestSuite()
        self.runner = RegressionTestRunner(
            config,
            progress_callback=self._progress_callback,
        )
        self.reporter = ReportGenerator(config.output_dir)
        
        # State
        self.results: List[TestResult] = []
        self.summary: Optional[EvalSummary] = None
        self._start_time: Optional[float] = None
    
    def _progress_callback(self, current: int, total: int, result: TestResult):
        """Progress callback for test execution."""
        status = "✅" if result.passed else "❌"
        logger.info(
            f"[{current}/{total}] {status} {result.test_id} | "
            f"Score: {result.scores.overall:.2f} | "
            f"Latency: {result.latency_ms:.0f}ms"
        )
    
    async def run(
        self,
        priority: Optional[TestPriority] = None,
        category: Optional[TestCategory] = None,
        parallel: bool = False,
    ) -> EvalSummary:
        """
        Run the evaluation pipeline.
        
        Args:
            priority: Optional priority filter
            category: Optional category filter
            parallel: Run tests in parallel
            
        Returns:
            EvalSummary with aggregated results
        """
        self._start_time = time.time()
        
        # Get test cases
        if priority == TestPriority.CRITICAL:
            test_cases = self.suite.get_critical_tests()
        elif priority:
            test_cases = self.suite.get_by_priority(priority)
        elif category:
            test_cases = self.suite.get_by_category(category)
        else:
            test_cases = self.suite.get_all_tests()
        
        logger.info(f"Running {len(test_cases)} tests against {self.config.model_id}")
        
        # Run tests
        if parallel:
            self.results = await self.runner.run_tests_parallel(
                test_cases,
                max_concurrent=self.config.batch_size,
            )
        else:
            self.results = await self.runner.run_tests(
                test_cases,
                priority_filter=priority,
            )
        
        # Compute summary
        self.summary = self.reporter.compute_summary(self.results, self.config)
        self.summary.total_duration_seconds = time.time() - self._start_time
        
        return self.summary
    
    def generate_report(
        self,
        summary: Optional[EvalSummary] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Generate evaluation reports.
        
        Args:
            summary: Optional summary (uses self.summary if not provided)
            output_dir: Optional output directory
            
        Returns:
            Dictionary with paths to generated files
        """
        if summary is None:
            summary = self.summary
        
        if summary is None:
            raise ValueError("No summary available. Run evaluation first.")
        
        if output_dir:
            self.reporter.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        # Save reports
        paths = self.reporter.save_reports(
            summary,
            self.results,
            prefix=f"eval_{self.config.model_id.replace('/', '_')}",
        )
        
        # Print summary to console
        self.reporter.print_summary(summary)
        
        logger.info(f"Reports saved to: {paths}")
        
        return paths
    
    async def compare_with_baseline(
        self,
        baseline_results: List[TestResult],
    ) -> dict:
        """
        Compare current results with baseline.
        
        Args:
            baseline_results: Results from baseline model
            
        Returns:
            Comparison metrics
        """
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        # Compute comparison
        current_score = sum(r.scores.overall for r in self.results) / len(self.results)
        baseline_score = sum(r.scores.overall for r in baseline_results) / len(baseline_results)
        
        improvement = current_score - baseline_score
        
        # Per-test comparison
        test_comparisons = []
        for current in self.results:
            baseline = next(
                (r for r in baseline_results if r.test_id == current.test_id),
                None
            )
            if baseline:
                test_comparisons.append({
                    "test_id": current.test_id,
                    "current_score": current.scores.overall,
                    "baseline_score": baseline.scores.overall,
                    "improvement": current.scores.overall - baseline.scores.overall,
                    "current_passed": current.passed,
                    "baseline_passed": baseline.passed,
                })
        
        return {
            "current_model": self.config.model_id,
            "baseline_model": self.config.baseline_model_id,
            "current_overall_score": current_score,
            "baseline_overall_score": baseline_score,
            "improvement": improvement,
            "improvement_percentage": (improvement / baseline_score * 100) if baseline_score > 0 else 0,
            "test_comparisons": test_comparisons,
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CognitiveTwin V3 Evaluation Pipeline"
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID to evaluate (Together AI or OpenAI format)",
    )
    parser.add_argument(
        "--baseline-model-id",
        default=None,
        help="Optional baseline model ID for comparison",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--priority",
        choices=["critical", "high", "medium", "low"],
        default=None,
        help="Filter tests by priority",
    )
    parser.add_argument(
        "--category",
        choices=["policy_compliance", "format_adherence", "content_quality", "behavioral_audit"],
        default=None,
        help="Filter tests by category",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for parallel execution",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all tests and exit",
    )
    
    args = parser.parse_args()
    
    # List tests mode
    if args.list_tests:
        suite = RegressionTestSuite()
        print("\nAvailable Tests:")
        print("=" * 60)
        for tc in suite.get_all_tests():
            print(f"  [{tc.priority.value:8}] {tc.test_id}: {tc.name}")
        print(f"\nTotal: {len(suite)} tests")
        return
    
    # Configure
    config = EvalConfig(
        model_id=args.model_id,
        baseline_model_id=args.baseline_model_id,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    
    # Parse priority
    priority = None
    if args.priority:
        priority = TestPriority(args.priority)
    
    # Parse category
    category = None
    if args.category:
        category = TestCategory(args.category)
    
    # Run pipeline
    pipeline = EvaluationPipeline(config)
    
    async def run_async():
        summary = await pipeline.run(
            priority=priority,
            category=category,
            parallel=args.parallel,
        )
        pipeline.generate_report()
        return summary
    
    try:
        summary = asyncio.run(run_async())
        
        # Exit code based on critical tests
        if summary.critical_pass_rate < 1.0:
            logger.warning("Some critical tests failed!")
            sys.exit(1)
        else:
            logger.info("All critical tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
