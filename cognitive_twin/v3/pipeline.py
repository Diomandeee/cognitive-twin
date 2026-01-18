"""
CognitiveTwin V3 Training Pipeline.

Implements Together AI fine-tuning for V3:
- Data preparation and upload
- Job management (SFT + DPO)
- Checkpoint evaluation
- A/B comparison

Usage:
    python -m rag_plusplus.ml.cognitivetwin_v3.pipeline train \
        --sft-data data/sft_train.jsonl \
        --dpo-data data/dpo_train.jsonl \
        --output-dir output/v3
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# BASE MODEL CONFIGURATION
# =============================================================================

BASE_MODELS = {
    "llama-3.1-8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "context_length": 128000,
        "supports_dpo": True,
        "supports_sft": True,
        "price_per_million_tokens": 0.18,
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "context_length": 128000,
        "supports_dpo": True,
        "supports_sft": True,
        "price_per_million_tokens": 0.06,
    },
    "qwen-2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "context_length": 32768,
        "supports_dpo": True,
        "supports_sft": True,
        "price_per_million_tokens": 0.27,
    },
}

DEFAULT_BASE_MODEL = "llama-3.1-8b"


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for Together AI training."""
    
    # Model
    base_model: str = DEFAULT_BASE_MODEL
    suffix: str = "cognitivetwin-v3"
    
    # Training mode
    training_type: str = "Full"  # "Full" or "LoRA"
    
    # Hyperparameters
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    warmup_ratio: float = 0.1
    
    # LoRA specific (if training_type == "LoRA")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # DPO specific
    dpo_beta: float = 0.1
    
    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 50
    
    # Data
    train_file_id: Optional[str] = None
    eval_file_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "model": BASE_MODELS[self.base_model]["model_id"] if self.base_model in BASE_MODELS else self.base_model,
            "suffix": self.suffix,
            "training_type": self.training_type,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "warmup_ratio": self.warmup_ratio,
            },
        }
        
        if self.training_type == "LoRA":
            result["lora_config"] = {
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            }
        
        return result


# =============================================================================
# TOGETHER AI CLIENT
# =============================================================================

class TogetherAIClient:
    """Client for Together AI fine-tuning."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self._client = None
    
    @property
    def client(self):
        """Lazy-load Together client."""
        if self._client is None:
            try:
                from together import Together
                self._client = Together(api_key=self.api_key)
            except ImportError:
                raise ImportError("together package not installed. Run: pip install together")
        return self._client
    
    def verify_connection(self) -> bool:
        """Verify API connection."""
        try:
            models = self.client.models.list()
            return len(models) > 0
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_model_info(self, model_key: str) -> dict:
        """Get model info from BASE_MODELS."""
        if model_key in BASE_MODELS:
            return BASE_MODELS[model_key]
        return {"model_id": model_key}


# =============================================================================
# DATA PREPARATION
# =============================================================================

class DataPreparer:
    """Prepare data for Together AI upload."""
    
    def prepare_sft_data(
        self,
        records: list,
        output_path: Path,
    ) -> Path:
        """Prepare SFT data in Together AI format."""
        formatted = []
        
        for record in records:
            # Handle both dict and dataclass
            if hasattr(record, 'to_dict'):
                record = record.to_dict()
            
            # Extract messages
            input_data = record.get("input", {})
            messages = input_data.get("messages", [])
            target = record.get("target", {}).get("assistant_content", "")
            
            # Format for chat completion training
            chat_messages = []
            
            for msg in messages:
                if hasattr(msg, 'to_dict'):
                    msg = msg.to_dict()
                chat_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            
            # Add target assistant message
            if target:
                chat_messages.append({
                    "role": "assistant",
                    "content": target,
                })
            
            if chat_messages:
                formatted.append({"messages": chat_messages})
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Prepared {len(formatted)} SFT records to {output_path}")
        return output_path
    
    def prepare_dpo_data(
        self,
        records: list,
        output_path: Path,
    ) -> Path:
        """Prepare DPO data in Together AI format."""
        formatted = []
        
        for record in records:
            # Handle both dict and dataclass
            if hasattr(record, 'to_dict'):
                record = record.to_dict()
            
            # Build prompt from messages
            input_data = record.get("input", {})
            messages = input_data.get("messages", [])
            
            prompt = ""
            for msg in messages:
                if hasattr(msg, 'to_dict'):
                    msg = msg.to_dict()
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            # Get preferred and dispreferred
            candidates = record.get("candidates", {})
            preferred = candidates.get("preferred", {}).get("assistant_content", "")
            dispreferred = candidates.get("dispreferred", {}).get("assistant_content", "")
            
            if prompt and preferred and dispreferred:
                formatted.append({
                    "prompt": prompt.strip(),
                    "chosen": preferred,
                    "rejected": dispreferred,
                })
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Prepared {len(formatted)} DPO records to {output_path}")
        return output_path


# =============================================================================
# DATA VALIDATION
# =============================================================================

class DataValidator:
    """Validate training data before upload."""
    
    def validate_sft_file(self, file_path: Path) -> tuple[bool, list[str]]:
        """Validate SFT data file."""
        errors = []
        line_count = 0
        
        with open(file_path) as f:
            for i, line in enumerate(f, 1):
                line_count += 1
                
                try:
                    data = json.loads(line)
                    
                    # Check for messages
                    if "messages" not in data:
                        errors.append(f"Line {i}: Missing 'messages' field")
                        continue
                    
                    messages = data["messages"]
                    
                    # Check message structure
                    for j, msg in enumerate(messages):
                        if "role" not in msg:
                            errors.append(f"Line {i}, msg {j}: Missing 'role'")
                        if "content" not in msg:
                            errors.append(f"Line {i}, msg {j}: Missing 'content'")
                    
                    # Check for at least one assistant message
                    has_assistant = any(m.get("role") == "assistant" for m in messages)
                    if not has_assistant:
                        errors.append(f"Line {i}: No assistant message")
                
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON - {e}")
        
        if line_count == 0:
            errors.append("File is empty")
        
        return len(errors) == 0, errors
    
    def validate_dpo_file(self, file_path: Path) -> tuple[bool, list[str]]:
        """Validate DPO data file."""
        errors = []
        line_count = 0
        
        with open(file_path) as f:
            for i, line in enumerate(f, 1):
                line_count += 1
                
                try:
                    data = json.loads(line)
                    
                    # Check required fields
                    if "prompt" not in data:
                        errors.append(f"Line {i}: Missing 'prompt' field")
                    if "chosen" not in data:
                        errors.append(f"Line {i}: Missing 'chosen' field")
                    if "rejected" not in data:
                        errors.append(f"Line {i}: Missing 'rejected' field")
                    
                    # Check content length
                    if len(data.get("chosen", "")) < 10:
                        errors.append(f"Line {i}: 'chosen' too short")
                    if len(data.get("rejected", "")) < 10:
                        errors.append(f"Line {i}: 'rejected' too short")
                
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON - {e}")
        
        if line_count == 0:
            errors.append("File is empty")
        
        return len(errors) == 0, errors


# =============================================================================
# DATA UPLOAD
# =============================================================================

class DataUploader:
    """Upload training data to Together AI."""
    
    def __init__(self, client: TogetherAIClient):
        self.client = client
    
    def upload_file(
        self,
        file_path: Path,
        purpose: str = "fine-tune",
    ) -> str:
        """Upload a file and return the file ID."""
        # Together AI expects a file path, not a file object
        file_path = Path(file_path)
        response = self.client.client.files.upload(
            file=str(file_path),
            purpose=purpose,
        )
        
        logger.info(f"Uploaded {file_path} -> {response.id}")
        return response.id
    
    def check_file_status(self, file_id: str) -> dict:
        """Check file processing status."""
        response = self.client.client.files.retrieve(file_id)
        return {
            "id": response.id,
            "status": getattr(response, "status", "unknown"),
            "filename": getattr(response, "filename", ""),
            "bytes": getattr(response, "bytes", 0),
        }
    
    def wait_for_processing(
        self,
        file_id: str,
        timeout: int = 300,
    ) -> bool:
        """Wait for file to finish processing."""
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.check_file_status(file_id)
            
            if status["status"] == "processed":
                logger.info(f"File {file_id} processed")
                return True
            elif status["status"] == "error":
                raise Exception(f"File processing failed: {file_id}")
            
            time.sleep(5)
        
        raise TimeoutError(f"File processing timeout: {file_id}")


# =============================================================================
# TRAINING JOB MANAGER
# =============================================================================

class TrainingJobManager:
    """Manage Together AI training jobs."""
    
    def __init__(self, client: TogetherAIClient):
        self.client = client
    
    def create_sft_job(
        self,
        config: TrainingConfig,
        train_file_id: str,
        eval_file_id: Optional[str] = None,
    ) -> str:
        """Create an SFT fine-tuning job."""
        model_id = (
            BASE_MODELS[config.base_model]["model_id"]
            if config.base_model in BASE_MODELS
            else config.base_model
        )
        
        job_config = {
            "model": model_id,
            "training_file": train_file_id,
            "suffix": config.suffix,
            "hyperparameters": {
                "learning_rate_multiplier": config.learning_rate / 1e-5,
                "n_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "warmup_ratio": config.warmup_ratio,
            },
        }
        
        if eval_file_id:
            job_config["validation_file"] = eval_file_id
        
        if config.training_type == "LoRA":
            job_config["training_type"] = "lora"
            job_config["lora_r"] = config.lora_r
            job_config["lora_alpha"] = config.lora_alpha
            job_config["lora_dropout"] = config.lora_dropout
        
        response = self.client.client.fine_tuning.create(**job_config)
        logger.info(f"Created SFT job: {response.id}")
        return response.id
    
    def create_dpo_job(
        self,
        config: TrainingConfig,
        train_file_id: str,
        eval_file_id: Optional[str] = None,
    ) -> str:
        """Create a DPO fine-tuning job."""
        model_id = (
            BASE_MODELS[config.base_model]["model_id"]
            if config.base_model in BASE_MODELS
            else config.base_model
        )
        
        job_config = {
            "model": model_id,
            "training_file": train_file_id,
            "suffix": f"{config.suffix}-dpo",
            "training_method": "dpo",
            "hyperparameters": {
                "learning_rate_multiplier": config.learning_rate / 1e-5,
                "n_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "dpo_beta": config.dpo_beta,
            },
        }
        
        if eval_file_id:
            job_config["validation_file"] = eval_file_id
        
        response = self.client.client.fine_tuning.create(**job_config)
        logger.info(f"Created DPO job: {response.id}")
        return response.id


# =============================================================================
# JOB MONITORING
# =============================================================================

class JobMonitor:
    """Monitor training job progress."""
    
    def __init__(self, client: TogetherAIClient):
        self.client = client
    
    def get_job_status(self, job_id: str) -> dict:
        """Get current job status."""
        response = self.client.client.fine_tuning.retrieve(job_id)
        
        return {
            "id": response.id,
            "status": response.status,
            "created_at": getattr(response, "created_at", None),
            "finished_at": getattr(response, "finished_at", None),
            "model": response.model,
            "fine_tuned_model": getattr(response, "fine_tuned_model", None),
            "training_file": response.training_file,
            "error": getattr(response, "error", None),
        }
    
    def get_job_events(self, job_id: str) -> list[dict]:
        """Get job training events."""
        response = self.client.client.fine_tuning.list_events(job_id)
        
        events = []
        for event in response.data:
            events.append({
                "created_at": getattr(event, "created_at", None),
                "level": getattr(event, "level", "info"),
                "message": getattr(event, "message", ""),
            })
        
        return events
    
    def get_training_metrics(self, job_id: str) -> dict:
        """Get training metrics from events."""
        events = self.get_job_events(job_id)
        
        metrics = {
            "steps": [],
            "train_loss": [],
            "eval_loss": [],
        }
        
        for event in events:
            msg = event.get("message", "")
            
            # Parse loss from message
            if "loss" in msg.lower():
                step_match = re.search(r"step[:\s]+(\d+)", msg, re.IGNORECASE)
                loss_match = re.search(r"loss[:\s]+([\d.]+)", msg, re.IGNORECASE)
                
                if step_match and loss_match:
                    metrics["steps"].append(int(step_match.group(1)))
                    metrics["train_loss"].append(float(loss_match.group(1)))
        
        return metrics
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 7200,
        callback: Optional[Callable] = None,
    ) -> dict:
        """Wait for job to complete."""
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.get_job_status(job_id)
            
            if callback:
                callback(status)
            
            if status["status"] == "succeeded":
                logger.info(f"Job {job_id} succeeded")
                return status
            elif status["status"] == "failed":
                raise Exception(f"Training failed: {status.get('error')}")
            elif status["status"] == "cancelled":
                raise Exception("Training was cancelled")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Training timeout after {timeout}s")


# =============================================================================
# JOB CONTROLLER
# =============================================================================

class JobController:
    """Control training jobs."""
    
    def __init__(self, client: TogetherAIClient):
        self.client = client
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        try:
            self.client.client.fine_tuning.cancel(job_id)
            logger.info(f"Cancelled job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    def list_jobs(self, limit: int = 10) -> list[dict]:
        """List recent jobs."""
        response = self.client.client.fine_tuning.list(limit=limit)
        
        jobs = []
        for job in response.data:
            jobs.append({
                "id": job.id,
                "status": job.status,
                "model": job.model,
                "created_at": getattr(job, "created_at", None),
                "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            })
        
        return jobs


# =============================================================================
# TRAINING STAGES
# =============================================================================

@dataclass
class StageResult:
    """Result of a training stage."""
    stage: str
    job_id: str
    model_id: str
    train_records: int
    eval_records: int
    duration_seconds: float = 0.0


class SFTTrainingStage:
    """SFT training stage."""
    
    def __init__(
        self,
        client: TogetherAIClient,
        config: TrainingConfig,
    ):
        self.client = client
        self.config = config
        self.preparer = DataPreparer()
        self.uploader = DataUploader(client)
        self.validator = DataValidator()
        self.job_manager = TrainingJobManager(client)
        self.monitor = JobMonitor(client)
    
    async def run(
        self,
        sft_records: list,
        output_dir: Path,
    ) -> StageResult:
        """Run SFT training stage."""
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        train_path = output_dir / "sft_train_upload.jsonl"
        eval_path = output_dir / "sft_eval_upload.jsonl"
        
        # Split data
        records = sft_records.copy()
        random.shuffle(records)
        split_idx = int(len(records) * 0.9)
        train_records = records[:split_idx]
        eval_records = records[split_idx:]
        
        # Prepare files
        self.preparer.prepare_sft_data(train_records, train_path)
        self.preparer.prepare_sft_data(eval_records, eval_path)
        
        # Validate
        valid_train, train_errors = self.validator.validate_sft_file(train_path)
        if not valid_train:
            raise Exception(f"Train data validation failed: {train_errors[:5]}")
        
        valid_eval, eval_errors = self.validator.validate_sft_file(eval_path)
        if not valid_eval:
            raise Exception(f"Eval data validation failed: {eval_errors[:5]}")
        
        # Upload
        train_file_id = self.uploader.upload_file(train_path)
        eval_file_id = self.uploader.upload_file(eval_path)
        
        # Wait for processing
        self.uploader.wait_for_processing(train_file_id)
        self.uploader.wait_for_processing(eval_file_id)
        
        # Create job
        job_id = self.job_manager.create_sft_job(
            self.config,
            train_file_id,
            eval_file_id,
        )
        
        # Monitor
        result = self.monitor.wait_for_completion(
            job_id,
            callback=lambda s: logger.info(f"SFT status: {s['status']}"),
        )
        
        duration = time.time() - start_time
        
        return StageResult(
            stage="sft",
            job_id=job_id,
            model_id=result["fine_tuned_model"],
            train_records=len(train_records),
            eval_records=len(eval_records),
            duration_seconds=duration,
        )


class DPOTrainingStage:
    """DPO training stage."""
    
    def __init__(
        self,
        client: TogetherAIClient,
        config: TrainingConfig,
    ):
        self.client = client
        self.config = config
        self.preparer = DataPreparer()
        self.uploader = DataUploader(client)
        self.validator = DataValidator()
        self.job_manager = TrainingJobManager(client)
        self.monitor = JobMonitor(client)
    
    async def run(
        self,
        dpo_records: list,
        base_model_id: str,
        output_dir: Path,
    ) -> StageResult:
        """Run DPO training stage."""
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with SFT model as base
        dpo_config = TrainingConfig(
            base_model=base_model_id,
            suffix=f"{self.config.suffix}-dpo",
            learning_rate=self.config.learning_rate / 2,  # Lower LR for DPO
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            dpo_beta=self.config.dpo_beta,
        )
        
        # Prepare data
        train_path = output_dir / "dpo_train_upload.jsonl"
        eval_path = output_dir / "dpo_eval_upload.jsonl"
        
        # Split data
        records = dpo_records.copy()
        random.shuffle(records)
        split_idx = int(len(records) * 0.9)
        train_records = records[:split_idx]
        eval_records = records[split_idx:]
        
        # Prepare files
        self.preparer.prepare_dpo_data(train_records, train_path)
        self.preparer.prepare_dpo_data(eval_records, eval_path)
        
        # Validate
        valid_train, train_errors = self.validator.validate_dpo_file(train_path)
        if not valid_train:
            raise Exception(f"DPO train data validation failed: {train_errors[:5]}")
        
        # Upload
        train_file_id = self.uploader.upload_file(train_path)
        eval_file_id = self.uploader.upload_file(eval_path)
        
        # Wait for processing
        self.uploader.wait_for_processing(train_file_id)
        self.uploader.wait_for_processing(eval_file_id)
        
        # Create job
        job_id = self.job_manager.create_dpo_job(
            dpo_config,
            train_file_id,
            eval_file_id,
        )
        
        # Monitor
        result = self.monitor.wait_for_completion(
            job_id,
            callback=lambda s: logger.info(f"DPO status: {s['status']}"),
        )
        
        duration = time.time() - start_time
        
        return StageResult(
            stage="dpo",
            job_id=job_id,
            model_id=result["fine_tuned_model"],
            train_records=len(train_records),
            eval_records=len(eval_records),
            duration_seconds=duration,
        )


# =============================================================================
# MODEL INFERENCE
# =============================================================================

class ModelInference:
    """Inference with trained model."""
    
    def __init__(self, client: TogetherAIClient, model_id: str):
        self.client = client
        self.model_id = model_id
    
    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate response from model."""
        response = self.client.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    
    def batch_generate(
        self,
        prompts: list[list[dict]],
        max_tokens: int = 2048,
    ) -> list[str]:
        """Batch generate responses."""
        responses = []
        for messages in prompts:
            try:
                response = self.generate(messages, max_tokens)
                responses.append(response)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                responses.append("")
        
        return responses


# =============================================================================
# REGRESSION TESTER
# =============================================================================

class RegressionTester:
    """Run regression tests on trained model."""
    
    def __init__(self, inference: ModelInference):
        self.inference = inference
    
    def run_eval_cases(self, eval_cases: list[dict]) -> dict:
        """Run evaluation cases."""
        results = {
            "total": len(eval_cases),
            "passed": 0,
            "failed": 0,
            "failures": [],
        }
        
        for case in eval_cases:
            # Build messages
            input_data = case.get("input", {})
            messages = []
            
            for msg in input_data.get("messages", []):
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            
            if not messages:
                continue
            
            # Generate response
            try:
                response = self.inference.generate(messages)
            except Exception as e:
                results["failed"] += 1
                results["failures"].append({
                    "case_id": case.get("record_id"),
                    "error": str(e),
                })
                continue
            
            # Check constraints
            checks = case.get("checks", {})
            passed, failures = self._check_constraints(response, checks)
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "case_id": case.get("record_id"),
                    "prompt": messages[-1]["content"][:100],
                    "response": response[:200],
                    "failures": failures,
                })
        
        if results["total"] > 0:
            results["pass_rate"] = results["passed"] / results["total"]
        else:
            results["pass_rate"] = 0.0
        
        return results
    
    def _check_constraints(
        self,
        response: str,
        checks: dict,
    ) -> tuple[bool, list[str]]:
        """Check response against constraints."""
        failures = []
        
        # Check disallowed phrases
        for phrase in checks.get("disallowed_phrases", []):
            if phrase.lower() in response.lower():
                failures.append(f"Contains disallowed phrase: '{phrase}'")
        
        # Check must not end with question
        if checks.get("must_not_end_with_question", False):
            if response.rstrip().endswith("?"):
                failures.append("Ends with question")
        
        # Check format
        if checks.get("must_follow_format") == "json":
            try:
                # Extract JSON from code block
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
                if json_match:
                    json.loads(json_match.group(1))
                else:
                    json.loads(response)
            except Exception:
                failures.append("Not valid JSON format")
        
        return len(failures) == 0, failures


# =============================================================================
# A/B COMPARISON
# =============================================================================

class ABComparison:
    """Compare V3 model against baseline."""
    
    def __init__(
        self,
        v3_model: ModelInference,
        baseline_model: ModelInference,
    ):
        self.v3 = v3_model
        self.baseline = baseline_model
    
    def compare_on_prompts(self, prompts: list[list[dict]]) -> dict:
        """Compare models on prompts."""
        results = {
            "v3_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
            "comparisons": [],
        }
        
        for messages in prompts:
            try:
                v3_response = self.v3.generate(messages)
                baseline_response = self.baseline.generate(messages)
            except Exception as e:
                logger.error(f"Comparison failed: {e}")
                continue
            
            # Score responses
            v3_score = self._score_response(v3_response, messages)
            baseline_score = self._score_response(baseline_response, messages)
            
            if v3_score > baseline_score:
                results["v3_wins"] += 1
                winner = "v3"
            elif baseline_score > v3_score:
                results["baseline_wins"] += 1
                winner = "baseline"
            else:
                results["ties"] += 1
                winner = "tie"
            
            results["comparisons"].append({
                "prompt": messages[-1]["content"][:100],
                "v3_response": v3_response[:200],
                "baseline_response": baseline_response[:200],
                "v3_score": v3_score,
                "baseline_score": baseline_score,
                "winner": winner,
            })
        
        total = len(prompts)
        if total > 0:
            results["v3_win_rate"] = results["v3_wins"] / total
            results["baseline_win_rate"] = results["baseline_wins"] / total
        else:
            results["v3_win_rate"] = 0.0
            results["baseline_win_rate"] = 0.0
        
        return results
    
    def _score_response(
        self,
        response: str,
        messages: list[dict],
    ) -> float:
        """Score a response (higher is better)."""
        score = 0.0
        
        # Penalize permission-seeking
        permission_phrases = [
            "would you like me to",
            "should i",
            "do you want me to",
            "can i proceed",
            "let me know if",
        ]
        
        response_lower = response.lower()
        for phrase in permission_phrases:
            if phrase in response_lower:
                score -= 0.3
        
        # Penalize ending with question
        if response.rstrip().endswith("?"):
            score -= 0.2
        
        # Reward code blocks (if expected)
        if "```" in response:
            score += 0.2
        
        # Reward directness (higher content density)
        words = len(response.split())
        if words > 50:  # Non-trivial response
            score += 0.1
        
        return score


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

@dataclass
class PipelineResult:
    """Result of training pipeline."""
    
    sft_model_id: str
    dpo_model_id: str
    sft_job_id: str
    dpo_job_id: str
    total_sft_records: int
    total_dpo_records: int
    training_time_seconds: float
    
    def to_dict(self) -> dict:
        return {
            "sft_model_id": self.sft_model_id,
            "dpo_model_id": self.dpo_model_id,
            "sft_job_id": self.sft_job_id,
            "dpo_job_id": self.dpo_job_id,
            "total_sft_records": self.total_sft_records,
            "total_dpo_records": self.total_dpo_records,
            "training_time_seconds": self.training_time_seconds,
        }


class V3TrainingPipeline:
    """Complete V3 training pipeline."""
    
    def __init__(
        self,
        together_api_key: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.client = TogetherAIClient(together_api_key)
        self.config = config or TrainingConfig()
    
    async def run(
        self,
        sft_records: list,
        dpo_records: list,
        output_dir: Path,
    ) -> PipelineResult:
        """Run complete training pipeline."""
        start_time = time.time()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: SFT
        logger.info("Starting SFT training stage...")
        sft_stage = SFTTrainingStage(self.client, self.config)
        sft_result = await sft_stage.run(sft_records, output_dir)
        
        logger.info(f"SFT complete: {sft_result.model_id}")
        
        # Stage 2: DPO
        logger.info("Starting DPO training stage...")
        dpo_stage = DPOTrainingStage(self.client, self.config)
        dpo_result = await dpo_stage.run(
            dpo_records,
            sft_result.model_id,
            output_dir,
        )
        
        logger.info(f"DPO complete: {dpo_result.model_id}")
        
        end_time = time.time()
        
        result = PipelineResult(
            sft_model_id=sft_result.model_id,
            dpo_model_id=dpo_result.model_id,
            sft_job_id=sft_result.job_id,
            dpo_job_id=dpo_result.job_id,
            total_sft_records=sft_result.train_records + sft_result.eval_records,
            total_dpo_records=dpo_result.train_records + dpo_result.eval_records,
            training_time_seconds=end_time - start_time,
        )
        
        # Save result
        with open(output_dir / "pipeline_result.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return result
    
    async def run_sft_only(
        self,
        sft_records: list,
        output_dir: Path,
    ) -> StageResult:
        """Run only SFT training."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sft_stage = SFTTrainingStage(self.client, self.config)
        return await sft_stage.run(sft_records, output_dir)
    
    async def run_dpo_only(
        self,
        dpo_records: list,
        base_model_id: str,
        output_dir: Path,
    ) -> StageResult:
        """Run only DPO training."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dpo_stage = DPOTrainingStage(self.client, self.config)
        return await dpo_stage.run(dpo_records, base_model_id, output_dir)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CognitiveTwin V3 Training Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--sft-data", type=Path, required=True, help="SFT data file")
    train_parser.add_argument("--dpo-data", type=Path, required=True, help="DPO data file")
    train_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    train_parser.add_argument("--base-model", default="llama-3.1-8b", help="Base model")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--suffix", default="cognitivetwin-v3", help="Model suffix")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model-id", required=True, help="Model ID to evaluate")
    eval_parser.add_argument("--eval-data", type=Path, required=True, help="Eval data file")
    eval_parser.add_argument("--output", type=Path, help="Output file for results")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="A/B compare models")
    compare_parser.add_argument("--v3-model", required=True, help="V3 model ID")
    compare_parser.add_argument("--baseline-model", required=True, help="Baseline model ID")
    compare_parser.add_argument("--prompts", type=Path, required=True, help="Prompts file")
    compare_parser.add_argument("--output", type=Path, help="Output file for results")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--job-id", help="Job ID to check")
    status_parser.add_argument("--list", action="store_true", help="List recent jobs")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if args.command == "train":
        # Load data
        sft_records = []
        dpo_records = []
        
        with open(args.sft_data) as f:
            for line in f:
                if line.strip():
                    sft_records.append(json.loads(line))
        
        with open(args.dpo_data) as f:
            for line in f:
                if line.strip():
                    dpo_records.append(json.loads(line))
        
        # Configure
        config = TrainingConfig(
            base_model=args.base_model,
            num_epochs=args.epochs,
            suffix=args.suffix,
        )
        
        # Run
        pipeline = V3TrainingPipeline(config=config)
        result = asyncio.run(pipeline.run(sft_records, dpo_records, args.output_dir))
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"SFT Model: {result.sft_model_id}")
        print(f"DPO Model: {result.dpo_model_id}")
        print(f"Training Time: {result.training_time_seconds:.1f}s")
        print("=" * 60)
    
    elif args.command == "evaluate":
        # Load eval cases
        eval_cases = []
        with open(args.eval_data) as f:
            for line in f:
                if line.strip():
                    eval_cases.append(json.loads(line))
        
        # Run evaluation
        client = TogetherAIClient()
        inference = ModelInference(client, args.model_id)
        tester = RegressionTester(inference)
        
        results = tester.run_eval_cases(eval_cases)
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Total: {results['total']}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print(f"Pass Rate: {results['pass_rate']:.2%}")
        print("=" * 60)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    
    elif args.command == "compare":
        # Load prompts
        prompts = []
        with open(args.prompts) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "messages" in data:
                        prompts.append(data["messages"])
        
        # Run comparison
        client = TogetherAIClient()
        v3_inference = ModelInference(client, args.v3_model)
        baseline_inference = ModelInference(client, args.baseline_model)
        
        comparison = ABComparison(v3_inference, baseline_inference)
        results = comparison.compare_on_prompts(prompts)
        
        print("\n" + "=" * 60)
        print("A/B Comparison Results")
        print("=" * 60)
        print(f"V3 Wins: {results['v3_wins']} ({results['v3_win_rate']:.2%})")
        print(f"Baseline Wins: {results['baseline_wins']} ({results['baseline_win_rate']:.2%})")
        print(f"Ties: {results['ties']}")
        print("=" * 60)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    
    elif args.command == "status":
        client = TogetherAIClient()
        
        if args.list:
            controller = JobController(client)
            jobs = controller.list_jobs()
            
            print("\n" + "=" * 60)
            print("Recent Jobs")
            print("=" * 60)
            for job in jobs:
                print(f"  {job['id']}: {job['status']} - {job['model']}")
            print("=" * 60)
        
        elif args.job_id:
            monitor = JobMonitor(client)
            status = monitor.get_job_status(args.job_id)
            
            print("\n" + "=" * 60)
            print(f"Job: {status['id']}")
            print("=" * 60)
            print(f"Status: {status['status']}")
            print(f"Model: {status['model']}")
            print(f"Fine-tuned Model: {status.get('fine_tuned_model', 'N/A')}")
            print("=" * 60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

