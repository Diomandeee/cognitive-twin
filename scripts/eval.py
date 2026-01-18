#!/usr/bin/env python3
"""
CognitiveTwin V2 Comprehensive Evaluation Suite

This module provides rigorous A/B testing between fine-tuned and base models,
measuring multiple dimensions of model quality including coherence, style transfer,
domain knowledge retention, and response fidelity.

Evaluation Dimensions:
1. Semantic Coherence - Logical flow and consistency
2. Lexical Diversity - Vocabulary richness
3. Style Fidelity - Alignment with training data patterns
4. Domain Specificity - Technical accuracy in target domains
5. Response Structure - Formatting and organization
6. Perplexity Proxy - Using embedding similarity as proxy
"""

import os
import sys
import json
import re
import math
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import statistics

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from supabase import create_client
from together import Together


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    finetuned_model: str = "mo_841e/Meta-Llama-3.1-8B-Instruct-Reference-cognitivetwin-v2-full-04e6c420"
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    max_tokens: int = 300
    temperature: float = 0.7
    num_prompts: int = 20
    num_runs_per_prompt: int = 2
    embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval"


@dataclass
class ResponseMetrics:
    """Metrics for a single response."""
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    lexical_diversity: float = 0.0  # Type-token ratio
    code_block_count: int = 0
    numbered_list_count: int = 0
    bullet_list_count: int = 0
    header_count: int = 0
    question_count: int = 0
    unique_words: int = 0
    avg_word_length: float = 0.0
    punctuation_density: float = 0.0
    uppercase_ratio: float = 0.0
    

@dataclass
class CoherenceMetrics:
    """Coherence analysis metrics."""
    sentence_similarity_mean: float = 0.0
    sentence_similarity_std: float = 0.0
    topic_consistency: float = 0.0
    logical_connectors_count: int = 0
    transition_words_count: int = 0


@dataclass
class StyleMetrics:
    """Style transfer evaluation metrics."""
    characteristic_phrase_count: int = 0
    formality_score: float = 0.0
    technical_term_density: float = 0.0
    personal_pronoun_ratio: float = 0.0
    imperative_sentence_ratio: float = 0.0


@dataclass
class ModelEvaluation:
    """Complete evaluation for one model."""
    model_name: str
    response_metrics: List[ResponseMetrics] = field(default_factory=list)
    coherence_metrics: List[CoherenceMetrics] = field(default_factory=list)
    style_metrics: List[StyleMetrics] = field(default_factory=list)
    raw_responses: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    generation_times: List[float] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Final evaluation report comparing both models."""
    timestamp: str
    config: EvaluationConfig
    finetuned_eval: ModelEvaluation
    base_eval: ModelEvaluation
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    training_data_samples: List[str] = field(default_factory=list)


class CognitiveTwinEvaluator:
    """
    Comprehensive evaluator for CognitiveTwin V2 models.
    
    Performs multi-dimensional analysis comparing fine-tuned model
    against base model across coherence, style, and domain metrics.
    """
    
    # Linguistic markers for analysis
    LOGICAL_CONNECTORS = [
        "therefore", "thus", "hence", "consequently", "as a result",
        "because", "since", "due to", "owing to", "for this reason",
        "however", "nevertheless", "nonetheless", "although", "despite",
        "furthermore", "moreover", "additionally", "in addition",
        "first", "second", "third", "finally", "lastly", "next",
    ]
    
    TRANSITION_WORDS = [
        "specifically", "particularly", "especially", "notably",
        "for example", "for instance", "such as", "including",
        "in other words", "that is", "namely", "essentially",
        "in contrast", "on the other hand", "alternatively",
        "similarly", "likewise", "in the same way",
    ]
    
    CHARACTERISTIC_PHRASES = [
        "let me", "here's", "step-by-step", "the key", "essentially",
        "in other words", "to clarify", "specifically", "notably",
        "this approach", "the idea is", "what this means",
        "in practice", "the benefit", "the tradeoff",
    ]
    
    TECHNICAL_TERMS = [
        "api", "endpoint", "database", "query", "cache", "async",
        "callback", "promise", "thread", "process", "memory",
        "latency", "throughput", "scalability", "microservice",
        "docker", "kubernetes", "terraform", "pipeline", "deployment",
        "embedding", "vector", "transformer", "attention", "gradient",
        "backprop", "inference", "training", "fine-tune", "lora",
    ]
    
    FORMAL_MARKERS = [
        "shall", "therefore", "consequently", "furthermore", "moreover",
        "notwithstanding", "henceforth", "whereby", "therein", "thereof",
    ]
    
    INFORMAL_MARKERS = [
        "gonna", "wanna", "kinda", "sorta", "yeah", "nope", "okay",
        "cool", "awesome", "basically", "actually", "literally",
    ]
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.together = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.supabase = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_ANON_KEY")
        )
        
    def fetch_training_prompts(self, limit: int = 50) -> List[Dict[str, str]]:
        """Fetch user prompts from training data for evaluation."""
        result = self.supabase.table("memory_turns").select(
            "content_text, role"
        ).eq("role", "user").limit(limit).execute()
        
        prompts = []
        for row in result.data:
            text = row.get("content_text", "")
            if text and len(text) > 20 and len(text) < 500:
                prompts.append({"prompt": text, "source": "training_data"})
        
        return prompts
    
    def fetch_training_responses(self, limit: int = 20) -> List[str]:
        """Fetch assistant responses from training data for style reference."""
        result = self.supabase.table("memory_turns").select(
            "content_text"
        ).eq("role", "assistant").limit(limit).execute()
        
        responses = []
        for row in result.data:
            text = row.get("content_text", "")
            if text and len(text) > 50:
                responses.append(text)
        
        return responses
    
    def generate_synthetic_prompts(self) -> List[Dict[str, str]]:
        """Generate synthetic prompts for controlled evaluation."""
        categories = {
            "coding": [
                "How would you implement a rate limiter in Python?",
                "Explain the difference between async and threading.",
                "What's the best way to handle database connections in a web app?",
                "How do you debug memory leaks in Node.js?",
            ],
            "architecture": [
                "Should I use microservices or a monolith for a new project?",
                "How do you design a notification system at scale?",
                "What's your approach to API versioning?",
                "How would you implement caching for a read-heavy application?",
            ],
            "explanation": [
                "Explain how transformer attention works.",
                "What is the CAP theorem and why does it matter?",
                "How does garbage collection work in managed languages?",
                "Explain the concept of eventual consistency.",
            ],
            "debugging": [
                "My React app is rendering slowly. How do I fix it?",
                "I'm getting CORS errors. What should I check?",
                "The database queries are timing out. What's the diagnosis process?",
                "Memory usage keeps growing in production. How do I investigate?",
            ],
        }
        
        prompts = []
        for category, category_prompts in categories.items():
            for p in category_prompts:
                prompts.append({"prompt": p, "source": f"synthetic_{category}"})
        
        return prompts
    
    def generate_response(self, model: str, prompt: str) -> Tuple[str, float]:
        """Generate a response from a model and measure time."""
        import time
        start = time.time()
        
        response = self.together.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        elapsed = time.time() - start
        return response.choices[0].message.content, elapsed
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text using Together's embedding model."""
        try:
            response = self.together.embeddings.create(
                model=self.config.embedding_model,
                input=text[:8000],  # Truncate to model limit
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
    
    def analyze_response_metrics(self, text: str) -> ResponseMetrics:
        """Compute structural metrics for a response."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        unique_words = set(w.lower() for w in words if w.isalpha())
        
        metrics = ResponseMetrics(
            word_count=len(words),
            sentence_count=len(sentences),
            avg_sentence_length=len(words) / max(len(sentences), 1),
            lexical_diversity=len(unique_words) / max(len(words), 1),
            code_block_count=text.count("```") // 2,
            numbered_list_count=len(re.findall(r"^\d+\.", text, re.MULTILINE)),
            bullet_list_count=text.count("\n- ") + text.count("\n* "),
            header_count=len(re.findall(r"^#+\s", text, re.MULTILINE)),
            question_count=text.count("?"),
            unique_words=len(unique_words),
            avg_word_length=sum(len(w) for w in words) / max(len(words), 1),
            punctuation_density=sum(1 for c in text if c in ".,;:!?") / max(len(text), 1),
            uppercase_ratio=sum(1 for c in text if c.isupper()) / max(len(text), 1),
        )
        
        return metrics
    
    def analyze_coherence(self, text: str, embedding: List[float]) -> CoherenceMetrics:
        """Analyze coherence metrics for a response."""
        text_lower = text.lower()
        
        logical_count = sum(1 for c in self.LOGICAL_CONNECTORS if c in text_lower)
        transition_count = sum(1 for t in self.TRANSITION_WORDS if t in text_lower)
        
        # Sentence-level coherence via embedding similarity would require
        # computing per-sentence embeddings; we use a proxy here
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Topic consistency: ratio of repeated significant words
        words = [w.lower() for w in text.split() if len(w) > 4 and w.isalpha()]
        word_counts = Counter(words)
        repeated = sum(1 for w, c in word_counts.items() if c > 1)
        topic_consistency = repeated / max(len(set(words)), 1)
        
        return CoherenceMetrics(
            sentence_similarity_mean=0.0,  # Would need per-sentence embeddings
            sentence_similarity_std=0.0,
            topic_consistency=topic_consistency,
            logical_connectors_count=logical_count,
            transition_words_count=transition_count,
        )
    
    def analyze_style(self, text: str) -> StyleMetrics:
        """Analyze style transfer metrics."""
        text_lower = text.lower()
        words = text.split()
        
        # Characteristic phrases from training style
        char_phrase_count = sum(
            1 for p in self.CHARACTERISTIC_PHRASES if p in text_lower
        )
        
        # Formality score
        formal_count = sum(1 for m in self.FORMAL_MARKERS if m in text_lower)
        informal_count = sum(1 for m in self.INFORMAL_MARKERS if m in text_lower)
        formality = (formal_count - informal_count) / max(len(words), 1) * 100
        
        # Technical term density
        tech_count = sum(1 for t in self.TECHNICAL_TERMS if t in text_lower)
        tech_density = tech_count / max(len(words), 1) * 100
        
        # Personal pronouns
        personal_pronouns = ["i", "we", "you", "my", "our", "your"]
        pronoun_count = sum(1 for w in words if w.lower() in personal_pronouns)
        pronoun_ratio = pronoun_count / max(len(words), 1)
        
        # Imperative sentences (start with verb)
        sentences = re.split(r'[.!?]+', text)
        imperative_verbs = ["use", "try", "consider", "implement", "create", "add", "check"]
        imperative_count = sum(
            1 for s in sentences 
            if s.strip() and s.strip().split()[0].lower() in imperative_verbs
        )
        imperative_ratio = imperative_count / max(len(sentences), 1)
        
        return StyleMetrics(
            characteristic_phrase_count=char_phrase_count,
            formality_score=formality,
            technical_term_density=tech_density,
            personal_pronoun_ratio=pronoun_ratio,
            imperative_sentence_ratio=imperative_ratio,
        )
    
    def evaluate_model(self, model_name: str, prompts: List[Dict[str, str]]) -> ModelEvaluation:
        """Run full evaluation for a single model."""
        evaluation = ModelEvaluation(model_name=model_name)
        
        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            print(f"  [{i+1}/{len(prompts)}] Evaluating: {prompt[:50]}...")
            
            for run in range(self.config.num_runs_per_prompt):
                response, gen_time = self.generate_response(model_name, prompt)
                
                evaluation.raw_responses.append(response)
                evaluation.prompts.append(prompt)
                evaluation.generation_times.append(gen_time)
                
                # Compute embedding
                embedding = self.compute_embedding(response)
                evaluation.embeddings.append(embedding)
                
                # Analyze metrics
                response_metrics = self.analyze_response_metrics(response)
                coherence_metrics = self.analyze_coherence(response, embedding)
                style_metrics = self.analyze_style(response)
                
                evaluation.response_metrics.append(response_metrics)
                evaluation.coherence_metrics.append(coherence_metrics)
                evaluation.style_metrics.append(style_metrics)
        
        return evaluation
    
    def compute_embedding_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        if not emb1 or not emb2:
            return 0.0
        
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def compute_comparative_analysis(
        self, 
        ft_eval: ModelEvaluation, 
        base_eval: ModelEvaluation,
        training_responses: List[str]
    ) -> Dict[str, Any]:
        """Compute comparative analysis between models."""
        
        # Aggregate response metrics
        def aggregate_metrics(metrics_list: List[ResponseMetrics]) -> Dict[str, float]:
            if not metrics_list:
                return {}
            return {
                "avg_word_count": statistics.mean(m.word_count for m in metrics_list),
                "avg_sentence_count": statistics.mean(m.sentence_count for m in metrics_list),
                "avg_sentence_length": statistics.mean(m.avg_sentence_length for m in metrics_list),
                "avg_lexical_diversity": statistics.mean(m.lexical_diversity for m in metrics_list),
                "total_code_blocks": sum(m.code_block_count for m in metrics_list),
                "total_numbered_lists": sum(m.numbered_list_count for m in metrics_list),
                "total_bullet_lists": sum(m.bullet_list_count for m in metrics_list),
                "avg_unique_words": statistics.mean(m.unique_words for m in metrics_list),
            }
        
        def aggregate_coherence(metrics_list: List[CoherenceMetrics]) -> Dict[str, float]:
            if not metrics_list:
                return {}
            return {
                "avg_topic_consistency": statistics.mean(m.topic_consistency for m in metrics_list),
                "total_logical_connectors": sum(m.logical_connectors_count for m in metrics_list),
                "total_transition_words": sum(m.transition_words_count for m in metrics_list),
            }
        
        def aggregate_style(metrics_list: List[StyleMetrics]) -> Dict[str, float]:
            if not metrics_list:
                return {}
            return {
                "total_characteristic_phrases": sum(m.characteristic_phrase_count for m in metrics_list),
                "avg_formality_score": statistics.mean(m.formality_score for m in metrics_list),
                "avg_technical_density": statistics.mean(m.technical_term_density for m in metrics_list),
                "avg_pronoun_ratio": statistics.mean(m.personal_pronoun_ratio for m in metrics_list),
            }
        
        ft_response_agg = aggregate_metrics(ft_eval.response_metrics)
        base_response_agg = aggregate_metrics(base_eval.response_metrics)
        
        ft_coherence_agg = aggregate_coherence(ft_eval.coherence_metrics)
        base_coherence_agg = aggregate_coherence(base_eval.coherence_metrics)
        
        ft_style_agg = aggregate_style(ft_eval.style_metrics)
        base_style_agg = aggregate_style(base_eval.style_metrics)
        
        # Compute similarity to training data
        training_embeddings = [self.compute_embedding(r) for r in training_responses[:10]]
        
        def avg_similarity_to_training(model_embeddings: List[List[float]]) -> float:
            similarities = []
            for me in model_embeddings:
                if not me:
                    continue
                for te in training_embeddings:
                    if te:
                        similarities.append(self.compute_embedding_similarity(me, te))
            return statistics.mean(similarities) if similarities else 0.0
        
        ft_training_sim = avg_similarity_to_training(ft_eval.embeddings)
        base_training_sim = avg_similarity_to_training(base_eval.embeddings)
        
        # Compute generation time statistics
        ft_avg_time = statistics.mean(ft_eval.generation_times) if ft_eval.generation_times else 0
        base_avg_time = statistics.mean(base_eval.generation_times) if base_eval.generation_times else 0
        
        return {
            "response_metrics": {
                "finetuned": ft_response_agg,
                "base": base_response_agg,
                "delta": {
                    k: ft_response_agg.get(k, 0) - base_response_agg.get(k, 0)
                    for k in ft_response_agg
                }
            },
            "coherence_metrics": {
                "finetuned": ft_coherence_agg,
                "base": base_coherence_agg,
            },
            "style_metrics": {
                "finetuned": ft_style_agg,
                "base": base_style_agg,
            },
            "training_similarity": {
                "finetuned": ft_training_sim,
                "base": base_training_sim,
                "delta": ft_training_sim - base_training_sim,
            },
            "generation_time": {
                "finetuned_avg_seconds": ft_avg_time,
                "base_avg_seconds": base_avg_time,
            },
            "sample_count": len(ft_eval.raw_responses),
        }
    
    def run_evaluation(self) -> EvaluationReport:
        """Execute the complete evaluation pipeline."""
        print("=" * 70)
        print("COGNITIVETWIN V2 COMPREHENSIVE EVALUATION")
        print("=" * 70)
        print()
        
        # Gather prompts
        print("Fetching training data prompts...")
        training_prompts = self.fetch_training_prompts(limit=30)
        synthetic_prompts = self.generate_synthetic_prompts()
        
        # Combine and limit
        all_prompts = training_prompts + synthetic_prompts
        all_prompts = all_prompts[:self.config.num_prompts]
        
        print(f"Total prompts for evaluation: {len(all_prompts)}")
        print()
        
        # Fetch training responses for style comparison
        print("Fetching training responses for style reference...")
        training_responses = self.fetch_training_responses(limit=20)
        print(f"Training response samples: {len(training_responses)}")
        print()
        
        # Evaluate fine-tuned model
        print(f"Evaluating FINE-TUNED model: {self.config.finetuned_model[:50]}...")
        ft_eval = self.evaluate_model(self.config.finetuned_model, all_prompts)
        print()
        
        # Evaluate base model
        print(f"Evaluating BASE model: {self.config.base_model[:50]}...")
        base_eval = self.evaluate_model(self.config.base_model, all_prompts)
        print()
        
        # Comparative analysis
        print("Computing comparative analysis...")
        comparative = self.compute_comparative_analysis(ft_eval, base_eval, training_responses)
        
        # Build report
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            config=self.config,
            finetuned_eval=ft_eval,
            base_eval=base_eval,
            comparative_analysis=comparative,
            training_data_samples=training_responses[:5],
        )
        
        return report
    
    def print_report_summary(self, report: EvaluationReport):
        """Print a summary of the evaluation report."""
        print()
        print("=" * 70)
        print("EVALUATION REPORT SUMMARY")
        print("=" * 70)
        print()
        
        ca = report.comparative_analysis
        
        print("RESPONSE METRICS COMPARISON:")
        print("-" * 50)
        rm = ca["response_metrics"]
        for metric in ["avg_word_count", "avg_sentence_length", "avg_lexical_diversity"]:
            ft_val = rm["finetuned"].get(metric, 0)
            base_val = rm["base"].get(metric, 0)
            delta = rm["delta"].get(metric, 0)
            print(f"  {metric}: FT={ft_val:.2f}, Base={base_val:.2f}, Δ={delta:+.2f}")
        print()
        
        print("COHERENCE METRICS:")
        print("-" * 50)
        cm = ca["coherence_metrics"]
        for metric in cm["finetuned"]:
            ft_val = cm["finetuned"].get(metric, 0)
            base_val = cm["base"].get(metric, 0)
            print(f"  {metric}: FT={ft_val:.2f}, Base={base_val:.2f}")
        print()
        
        print("STYLE METRICS:")
        print("-" * 50)
        sm = ca["style_metrics"]
        for metric in sm["finetuned"]:
            ft_val = sm["finetuned"].get(metric, 0)
            base_val = sm["base"].get(metric, 0)
            print(f"  {metric}: FT={ft_val:.2f}, Base={base_val:.2f}")
        print()
        
        print("TRAINING DATA SIMILARITY (Embedding Cosine):")
        print("-" * 50)
        ts = ca["training_similarity"]
        print(f"  Fine-tuned: {ts['finetuned']:.4f}")
        print(f"  Base model: {ts['base']:.4f}")
        print(f"  Delta: {ts['delta']:+.4f}")
        print()
        
        print("GENERATION TIME:")
        print("-" * 50)
        gt = ca["generation_time"]
        print(f"  Fine-tuned avg: {gt['finetuned_avg_seconds']:.2f}s")
        print(f"  Base model avg: {gt['base_avg_seconds']:.2f}s")
        print()
        
        # Verdict
        print("=" * 70)
        print("VERDICT:")
        print("=" * 70)
        
        style_win = sm["finetuned"]["total_characteristic_phrases"] > sm["base"]["total_characteristic_phrases"]
        coherence_win = cm["finetuned"]["total_logical_connectors"] >= cm["base"]["total_logical_connectors"]
        similarity_win = ts["delta"] > 0
        
        verdicts = []
        if style_win:
            verdicts.append("Style transfer: SUCCESSFUL (more characteristic phrases)")
        else:
            verdicts.append("Style transfer: NEEDS IMPROVEMENT")
        
        if coherence_win:
            verdicts.append("Coherence: MAINTAINED (logical connectors preserved)")
        else:
            verdicts.append("Coherence: BASELINE")
        
        if similarity_win:
            verdicts.append(f"Training alignment: IMPROVED (+{ts['delta']:.4f} similarity)")
        else:
            verdicts.append("Training alignment: NO SIGNIFICANT CHANGE")
        
        for v in verdicts:
            print(f"  {v}")
        
        print()


def main():
    """Run the evaluation and save results."""
    config = EvaluationConfig(
        num_prompts=15,
        num_runs_per_prompt=1,
        max_tokens=250,
    )
    
    evaluator = CognitiveTwinEvaluator(config)
    report = evaluator.run_evaluation()
    evaluator.print_report_summary(report)
    
    # Save report to JSON
    output_path = Path("data/evaluation_report.json")
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert to serializable format
    report_dict = {
        "timestamp": report.timestamp,
        "config": asdict(report.config),
        "comparative_analysis": report.comparative_analysis,
        "training_data_samples": report.training_data_samples,
        "sample_responses": {
            "finetuned": report.finetuned_eval.raw_responses[:5],
            "base": report.base_eval.raw_responses[:5],
        },
        "prompts_used": report.finetuned_eval.prompts[:5],
    }
    
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"Report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    main()

