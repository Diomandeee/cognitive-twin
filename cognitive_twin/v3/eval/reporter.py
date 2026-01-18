"""
Report Generator for CognitiveTwin V3 Evaluation.

Generates markdown and JSON reports from evaluation results.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from .types import (
    EvalSummary,
    TestResult,
    TestCategory,
    TestPriority,
    CategoryMetrics,
    EvalConfig,
)


class ReportGenerator:
    """
    Generate evaluation reports.
    
    Produces:
    - Markdown summary report
    - JSON detailed results
    - Category breakdown
    - Failure analysis
    """
    
    def __init__(self, output_dir: str = "eval_results"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_summary(
        self,
        results: List[TestResult],
        config: EvalConfig,
    ) -> EvalSummary:
        """
        Compute evaluation summary from results.
        
        Args:
            results: List of test results
            config: Evaluation configuration
            
        Returns:
            EvalSummary with aggregated metrics
        """
        # Basic counts
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        # Average scores
        if results:
            overall_score = sum(r.scores.overall for r in results) / total
            policy_score = sum(r.scores.policy_compliance for r in results) / total
            format_score = sum(r.scores.format_adherence for r in results) / total
            content_score = sum(r.scores.content_quality for r in results) / total
            avg_latency = sum(r.latency_ms for r in results) / total
        else:
            overall_score = policy_score = format_score = content_score = 0.0
            avg_latency = 0.0
        
        # Category breakdown
        category_metrics = {}
        for category in TestCategory:
            cat_results = [r for r in results if r.category == category]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.passed)
                category_metrics[category] = CategoryMetrics(
                    category=category,
                    total_tests=len(cat_results),
                    passed_tests=cat_passed,
                    failed_tests=len(cat_results) - cat_passed,
                    avg_score=sum(r.scores.overall for r in cat_results) / len(cat_results),
                    avg_latency_ms=sum(r.latency_ms for r in cat_results) / len(cat_results),
                )
        
        # Priority breakdown
        critical_results = [r for r in results if r.priority == TestPriority.CRITICAL]
        critical_pass_rate = (
            sum(1 for r in critical_results if r.passed) / len(critical_results)
            if critical_results else 0.0
        )
        
        high_results = [r for r in results if r.priority == TestPriority.HIGH]
        high_pass_rate = (
            sum(1 for r in high_results if r.passed) / len(high_results)
            if high_results else 0.0
        )
        
        return EvalSummary(
            model_id=config.model_id,
            baseline_model_id=config.baseline_model_id,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            overall_score=overall_score,
            policy_compliance_score=policy_score,
            format_adherence_score=format_score,
            content_quality_score=content_score,
            avg_latency_ms=avg_latency,
            category_metrics=category_metrics,
            critical_pass_rate=critical_pass_rate,
            high_pass_rate=high_pass_rate,
            config=config,
        )
    
    def generate_markdown_report(
        self,
        summary: EvalSummary,
        results: List[TestResult],
    ) -> str:
        """
        Generate a markdown report.
        
        Args:
            summary: Evaluation summary
            results: List of test results
            
        Returns:
            Markdown report string
        """
        lines = []
        
        # Header
        lines.append("# CognitiveTwin V3 Evaluation Report")
        lines.append("")
        lines.append(f"**Model:** {summary.model_id}")
        if summary.baseline_model_id:
            lines.append(f"**Baseline:** {summary.baseline_model_id}")
        lines.append(f"**Timestamp:** {summary.timestamp.isoformat()}")
        lines.append("")
        
        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Tests | {summary.total_tests} |")
        lines.append(f"| Passed | {summary.passed_tests} |")
        lines.append(f"| Failed | {summary.failed_tests} |")
        lines.append(f"| Pass Rate | {summary.pass_rate:.1%} |")
        lines.append(f"| Overall Score | {summary.overall_score:.2f} |")
        lines.append(f"| Avg Latency | {summary.avg_latency_ms:.0f}ms |")
        lines.append("")
        
        # Score breakdown
        lines.append("## Score Breakdown")
        lines.append("")
        lines.append(f"| Category | Score |")
        lines.append("|----------|-------|")
        lines.append(f"| Policy Compliance | {summary.policy_compliance_score:.2f} |")
        lines.append(f"| Format Adherence | {summary.format_adherence_score:.2f} |")
        lines.append(f"| Content Quality | {summary.content_quality_score:.2f} |")
        lines.append("")
        
        # Priority breakdown
        lines.append("## Priority Breakdown")
        lines.append("")
        lines.append(f"| Priority | Pass Rate |")
        lines.append("|----------|-----------|")
        lines.append(f"| Critical | {summary.critical_pass_rate:.1%} |")
        lines.append(f"| High | {summary.high_pass_rate:.1%} |")
        lines.append("")
        
        # Category breakdown
        lines.append("## Category Breakdown")
        lines.append("")
        lines.append(f"| Category | Tests | Passed | Failed | Pass Rate | Avg Score |")
        lines.append("|----------|-------|--------|--------|-----------|-----------|")
        for category, metrics in summary.category_metrics.items():
            lines.append(
                f"| {category.value} | {metrics.total_tests} | "
                f"{metrics.passed_tests} | {metrics.failed_tests} | "
                f"{metrics.pass_rate:.1%} | {metrics.avg_score:.2f} |"
            )
        lines.append("")
        
        # Failed tests
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            lines.append("## Failed Tests")
            lines.append("")
            
            for result in failed_results:
                lines.append(f"### {result.test_id}: {result.test_name}")
                lines.append("")
                lines.append(f"- **Priority:** {result.priority.value}")
                lines.append(f"- **Category:** {result.category.value}")
                lines.append(f"- **Score:** {result.scores.overall:.2f}")
                lines.append("")
                
                if result.failures:
                    lines.append("**Failures:**")
                    for failure in result.failures[:5]:  # Limit to 5
                        lines.append(f"- {failure.message}")
                    lines.append("")
                
                lines.append("**Response (truncated):**")
                lines.append("```")
                lines.append(result.response[:500] + ("..." if len(result.response) > 500 else ""))
                lines.append("```")
                lines.append("")
        
        # Passed tests summary
        passed_results = [r for r in results if r.passed]
        if passed_results:
            lines.append("## Passed Tests")
            lines.append("")
            lines.append(f"| Test ID | Name | Score | Latency |")
            lines.append("|---------|------|-------|---------|")
            for result in passed_results:
                lines.append(
                    f"| {result.test_id} | {result.test_name[:30]}... | "
                    f"{result.scores.overall:.2f} | {result.latency_ms:.0f}ms |"
                )
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_json_report(
        self,
        summary: EvalSummary,
        results: List[TestResult],
    ) -> dict:
        """
        Generate a JSON report.
        
        Args:
            summary: Evaluation summary
            results: List of test results
            
        Returns:
            Dictionary with full report data
        """
        return {
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results],
            "generated_at": datetime.now().isoformat(),
        }
    
    def save_reports(
        self,
        summary: EvalSummary,
        results: List[TestResult],
        prefix: str = "eval",
    ) -> Dict[str, str]:
        """
        Save reports to files.
        
        Args:
            summary: Evaluation summary
            results: List of test results
            prefix: Filename prefix
            
        Returns:
            Dictionary with paths to generated files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate reports
        markdown = self.generate_markdown_report(summary, results)
        json_report = self.generate_json_report(summary, results)
        
        # Save files
        md_path = os.path.join(self.output_dir, f"{prefix}_{timestamp}.md")
        json_path = os.path.join(self.output_dir, f"{prefix}_{timestamp}.json")
        
        with open(md_path, "w") as f:
            f.write(markdown)
        
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2)
        
        return {
            "markdown": md_path,
            "json": json_path,
        }
    
    def print_summary(self, summary: EvalSummary):
        """
        Print a summary to console.
        
        Args:
            summary: Evaluation summary
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {summary.model_id}")
        print(f"Total Tests: {summary.total_tests}")
        print(f"Passed: {summary.passed_tests} ({summary.pass_rate:.1%})")
        print(f"Failed: {summary.failed_tests}")
        print("-" * 60)
        print(f"Overall Score: {summary.overall_score:.2f}")
        print(f"Policy Compliance: {summary.policy_compliance_score:.2f}")
        print(f"Format Adherence: {summary.format_adherence_score:.2f}")
        print(f"Content Quality: {summary.content_quality_score:.2f}")
        print("-" * 60)
        print(f"Critical Pass Rate: {summary.critical_pass_rate:.1%}")
        print(f"High Priority Pass Rate: {summary.high_pass_rate:.1%}")
        print(f"Avg Latency: {summary.avg_latency_ms:.0f}ms")
        print("=" * 60)
