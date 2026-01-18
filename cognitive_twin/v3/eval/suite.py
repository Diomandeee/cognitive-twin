"""
Regression Test Suite for CognitiveTwin V3 Evaluation.

Aggregates all test generators and provides filtering utilities.
"""

from typing import List, Optional

from .types import (
    TestCase,
    TestCategory,
    TestPriority,
)
from .test_cases import (
    QuestionPolicyTests,
    FormatComplianceTests,
    OmissionTests,
    HistoricalAnnoyanceCases,
    EdgeCaseTests,
)


class RegressionTestSuite:
    """
    Complete regression test suite.
    
    Aggregates all test generators and provides filtering utilities.
    
    Usage:
        suite = RegressionTestSuite()
        
        # Get all tests
        all_tests = suite.get_all_tests()
        
        # Get critical tests only
        critical_tests = suite.get_critical_tests()
        
        # Get by category
        policy_tests = suite.get_by_category(TestCategory.POLICY_COMPLIANCE)
    """
    
    def __init__(self):
        """Initialize the test suite."""
        self._test_generators = [
            QuestionPolicyTests,
            FormatComplianceTests,
            OmissionTests,
            HistoricalAnnoyanceCases,
            EdgeCaseTests,
        ]
        
        # Cache for all tests
        self._all_tests: Optional[List[TestCase]] = None
    
    def get_all_tests(self) -> List[TestCase]:
        """
        Get all test cases from all generators.
        
        Returns:
            List of all TestCase objects
        """
        if self._all_tests is None:
            self._all_tests = []
            for generator in self._test_generators:
                self._all_tests.extend(generator.get_all())
        
        return self._all_tests
    
    def get_critical_tests(self) -> List[TestCase]:
        """
        Get only critical priority tests.
        
        These tests must pass for model deployment.
        
        Returns:
            List of critical TestCase objects
        """
        return [
            tc for tc in self.get_all_tests()
            if tc.priority == TestPriority.CRITICAL
        ]
    
    def get_high_priority_tests(self) -> List[TestCase]:
        """
        Get critical and high priority tests.
        
        Returns:
            List of critical and high priority TestCase objects
        """
        return [
            tc for tc in self.get_all_tests()
            if tc.priority in [TestPriority.CRITICAL, TestPriority.HIGH]
        ]
    
    def get_by_category(self, category: TestCategory) -> List[TestCase]:
        """
        Get tests by category.
        
        Args:
            category: The test category to filter by
            
        Returns:
            List of TestCase objects in the specified category
        """
        return [
            tc for tc in self.get_all_tests()
            if tc.category == category
        ]
    
    def get_by_priority(self, priority: TestPriority) -> List[TestCase]:
        """
        Get tests by priority.
        
        Args:
            priority: The priority level to filter by
            
        Returns:
            List of TestCase objects with the specified priority
        """
        return [
            tc for tc in self.get_all_tests()
            if tc.priority == priority
        ]
    
    def get_by_tag(self, tag: str) -> List[TestCase]:
        """
        Get tests by tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of TestCase objects with the specified tag
        """
        return [
            tc for tc in self.get_all_tests()
            if tag in tc.tags
        ]
    
    def get_by_source(self, source: str) -> List[TestCase]:
        """
        Get tests by source.
        
        Args:
            source: The source to filter by (e.g., "synthetic", "friction")
            
        Returns:
            List of TestCase objects from the specified source
        """
        return [
            tc for tc in self.get_all_tests()
            if tc.source == source
        ]
    
    def get_test_by_id(self, test_id: str) -> Optional[TestCase]:
        """
        Get a specific test by ID.
        
        Args:
            test_id: The test ID to look up
            
        Returns:
            TestCase if found, None otherwise
        """
        for tc in self.get_all_tests():
            if tc.test_id == test_id:
                return tc
        return None
    
    def get_summary(self) -> dict:
        """
        Get a summary of the test suite.
        
        Returns:
            Dictionary with test counts by category and priority
        """
        all_tests = self.get_all_tests()
        
        by_category = {}
        for category in TestCategory:
            by_category[category.value] = len([
                tc for tc in all_tests if tc.category == category
            ])
        
        by_priority = {}
        for priority in TestPriority:
            by_priority[priority.value] = len([
                tc for tc in all_tests if tc.priority == priority
            ])
        
        return {
            "total_tests": len(all_tests),
            "by_category": by_category,
            "by_priority": by_priority,
            "generators": [g.__name__ for g in self._test_generators],
        }
    
    def __len__(self) -> int:
        """Return total number of tests."""
        return len(self.get_all_tests())
    
    def __iter__(self):
        """Iterate over all tests."""
        return iter(self.get_all_tests())
