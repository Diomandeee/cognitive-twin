"""
Test Runner for CognitiveTwin V3 Evaluation.

Executes test cases against models and collects results.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .types import (
    EvalConfig,
    TestCase,
    TestResult,
    ScoreBreakdown,
    FailureDetail,
    TestPriority,
)
from .scorers import (
    PolicyComplianceScorer,
    FormatAdherenceScorer,
    ContentQualityScorer,
)

logger = logging.getLogger(__name__)


class ModelClient:
    """Client for model inference."""
    
    def __init__(self, config: EvalConfig):
        """Initialize model client."""
        self.config = config
        self._client = None
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            
        Returns:
            Model response text
        """
        # Try Together AI first
        try:
            return await self._generate_together(messages, system_prompt)
        except Exception as e:
            logger.warning(f"Together AI failed: {e}, trying OpenAI...")
        
        # Fallback to OpenAI
        try:
            return await self._generate_openai(messages, system_prompt)
        except Exception as e:
            logger.error(f"All generation attempts failed: {e}")
            raise
    
    async def _generate_together(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate using Together AI."""
        import os
        
        try:
            from together import Together
        except ImportError:
            raise ImportError("together package not installed")
        
        api_key = self.config.api_key or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not set")
        
        client = Together(api_key=api_key)
        
        # Build messages
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=self.config.model_id,
            messages=full_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return response.choices[0].message.content
    
    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate using OpenAI."""
        import os
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed")
        
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        client = OpenAI(api_key=api_key)
        
        # Build messages
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=self.config.model_id,
            messages=full_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return response.choices[0].message.content


class RegressionTestRunner:
    """Run regression tests against a model."""
    
    def __init__(
        self,
        config: EvalConfig,
        progress_callback: Optional[Callable[[int, int, TestResult], None]] = None,
    ):
        """
        Initialize test runner.
        
        Args:
            config: Evaluation configuration
            progress_callback: Optional callback for progress updates
                               (current, total, result)
        """
        self.config = config
        self.progress_callback = progress_callback
        
        # Initialize scorers
        self.policy_scorer = PolicyComplianceScorer()
        self.format_scorer = FormatAdherenceScorer()
        self.content_scorer = ContentQualityScorer()
        
        # Initialize model client
        self.client = ModelClient(config)
    
    async def run_test(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: The test case to run
            
        Returns:
            TestResult with scores and failures
        """
        start_time = time.time()
        
        # Convert messages to API format
        messages = [m.to_dict() for m in test_case.messages]
        
        # Generate response
        try:
            response = await self.client.generate(
                messages=messages,
                system_prompt=test_case.system_prompt,
            )
        except Exception as e:
            logger.error(f"Generation failed for {test_case.test_id}: {e}")
            return TestResult(
                test_id=test_case.test_id,
                test_name=test_case.name,
                category=test_case.category,
                priority=test_case.priority,
                passed=False,
                scores=ScoreBreakdown(),
                response=f"ERROR: {e}",
                latency_ms=(time.time() - start_time) * 1000,
                failures=[FailureDetail(
                    check_type="generation",
                    expected="Valid response",
                    actual=str(e),
                    message=f"Model generation failed: {e}",
                )],
                model_id=self.config.model_id,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Score the response
        all_failures = []
        
        # Policy compliance
        policy_score, policy_failures = self.policy_scorer.score(response, test_case)
        all_failures.extend(policy_failures)
        
        # Format adherence
        format_score, format_failures = self.format_scorer.score(response, test_case)
        all_failures.extend(format_failures)
        
        # Content quality
        content_score, content_failures = self.content_scorer.score(response, test_case)
        all_failures.extend(content_failures)
        
        # Compute overall scores
        scores = ScoreBreakdown(
            policy_compliance=policy_score.overall,
            format_adherence=format_score.overall,
            content_quality=content_score.overall,
            behavioral_score=1.0 if len(all_failures) == 0 else max(0, 1.0 - len(all_failures) * 0.1),
        )
        
        # Determine pass/fail
        # Critical tests must have no failures
        # Other tests can have partial failures
        if test_case.priority == TestPriority.CRITICAL:
            passed = len(all_failures) == 0
        else:
            passed = scores.overall >= 0.7
        
        return TestResult(
            test_id=test_case.test_id,
            test_name=test_case.name,
            category=test_case.category,
            priority=test_case.priority,
            passed=passed,
            scores=scores,
            response=response,
            latency_ms=latency_ms,
            failures=all_failures,
            timestamp=datetime.now(),
            model_id=self.config.model_id,
        )
    
    async def run_tests(
        self,
        test_cases: List[TestCase],
        priority_filter: Optional[TestPriority] = None,
    ) -> List[TestResult]:
        """
        Run multiple test cases.
        
        Args:
            test_cases: List of test cases to run
            priority_filter: Optional filter by priority
            
        Returns:
            List of TestResults
        """
        # Filter by priority if specified
        if priority_filter:
            test_cases = [
                tc for tc in test_cases
                if tc.priority == priority_filter or (
                    priority_filter == TestPriority.HIGH and 
                    tc.priority in [TestPriority.CRITICAL, TestPriority.HIGH]
                )
            ]
        
        results = []
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            if self.config.verbose:
                logger.info(f"Running test {i+1}/{total}: {test_case.test_id}")
            
            result = await self.run_test(test_case)
            results.append(result)
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback(i + 1, total, result)
            
            # Log result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            if self.config.verbose:
                logger.info(
                    f"  {status} | Score: {result.scores.overall:.2f} | "
                    f"Latency: {result.latency_ms:.0f}ms"
                )
        
        return results
    
    async def run_tests_parallel(
        self,
        test_cases: List[TestCase],
        max_concurrent: int = 5,
    ) -> List[TestResult]:
        """
        Run multiple test cases in parallel.
        
        Args:
            test_cases: List of test cases to run
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of TestResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def run_with_semaphore(test_case: TestCase) -> TestResult:
            async with semaphore:
                return await self.run_test(test_case)
        
        tasks = [run_with_semaphore(tc) for tc in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_case = test_cases[i]
                final_results.append(TestResult(
                    test_id=test_case.test_id,
                    test_name=test_case.name,
                    category=test_case.category,
                    priority=test_case.priority,
                    passed=False,
                    scores=ScoreBreakdown(),
                    response=f"ERROR: {result}",
                    latency_ms=0,
                    failures=[FailureDetail(
                        check_type="execution",
                        expected="Successful execution",
                        actual=str(result),
                        message=f"Test execution failed: {result}",
                    )],
                    model_id=self.config.model_id,
                ))
            else:
                final_results.append(result)
        
        return final_results
