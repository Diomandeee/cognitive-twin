# Phase 4: Training Pipeline

> **Purpose**: Configure and execute Together AI DPO training for CognitiveTwin V3, including data upload, job management, and checkpoint evaluation.
>
> **Platform**: Together AI Fine-tuning API
>
> **Implementation File**: `rag_plusplus/ml/cognitivetwin_v3/pipeline.py`

---

## 1. Together AI Configuration

### 1.1. API Setup

```python
from together import Together
import os

class TogetherAIClient:
    """Client for Together AI fine-tuning."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.client = Together(api_key=self.api_key)
    
    def verify_connection(self) -> bool:
        """Verify API connection."""
        try:
            models = self.client.models.list()
            return len(models) > 0
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
```

### 1.2. Base Model Selection

```python
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
```

### 1.3. Training Configuration

```python
from dataclasses import dataclass, field
from typing import Optional

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
        return {
            "model": BASE_MODELS[self.base_model]["model_id"],
            "suffix": self.suffix,
            "training_type": self.training_type,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "warmup_ratio": self.warmup_ratio,
            },
            "lora_config": {
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            } if self.training_type == "LoRA" else None,
        }
```

---

## 2. Data Upload

### 2.1. File Preparation

```python
import json
from pathlib import Path

class DataPreparer:
    """Prepare data for Together AI upload."""
    
    def prepare_sft_data(
        self,
        records: list,
        output_path: Path
    ) -> Path:
        """Prepare SFT data in Together AI format."""
        
        formatted = []
        
        for record in records:
            # Extract messages
            messages = record["input"]["messages"]
            target = record["target"]["assistant_content"]
            
            # Format for chat completion training
            chat_messages = []
            
            for msg in messages:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            
            # Add target assistant message
            chat_messages.append({
                "role": "assistant",
                "content": target,
            })
            
            formatted.append({"messages": chat_messages})
        
        # Write to file
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        
        return output_path
    
    def prepare_dpo_data(
        self,
        records: list,
        output_path: Path
    ) -> Path:
        """Prepare DPO data in Together AI format."""
        
        formatted = []
        
        for record in records:
            # Build prompt from messages
            messages = record["input"]["messages"]
            
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"
            
            # Get preferred and dispreferred
            preferred = record["candidates"]["preferred"]["assistant_content"]
            dispreferred = record["candidates"]["dispreferred"]["assistant_content"]
            
            formatted.append({
                "prompt": prompt.strip(),
                "chosen": preferred,
                "rejected": dispreferred,
            })
        
        # Write to file
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        
        return output_path
```

### 2.2. File Upload

```python
class DataUploader:
    """Upload training data to Together AI."""
    
    def __init__(self, client: TogetherAIClient):
        self.client = client
    
    def upload_file(
        self,
        file_path: Path,
        purpose: str = "fine-tune"
    ) -> str:
        """Upload a file and return the file ID."""
        
        with open(file_path, 'rb') as f:
            response = self.client.client.files.upload(
                file=f,
                purpose=purpose,
            )
        
        return response.id
    
    def check_file_status(self, file_id: str) -> dict:
        """Check file processing status."""
        
        response = self.client.client.files.retrieve(file_id)
        return {
            "id": response.id,
            "status": response.status,
            "filename": response.filename,
            "bytes": response.bytes,
        }
    
    def wait_for_processing(
        self,
        file_id: str,
        timeout: int = 300
    ) -> bool:
        """Wait for file to finish processing."""
        
        import time
        
        start = time.time()
        while time.time() - start < timeout:
            status = self.check_file_status(file_id)
            
            if status["status"] == "processed":
                return True
            elif status["status"] == "error":
                raise Exception(f"File processing failed: {file_id}")
            
            time.sleep(5)
        
        raise TimeoutError(f"File processing timeout: {file_id}")
```

### 2.3. Data Validation

```python
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
                    has_assistant = any(m["role"] == "assistant" for m in messages)
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
```

---

## 3. Training Job Management

### 3.1. Job Creation

```python
class TrainingJobManager:
    """Manage Together AI training jobs."""
    
    def __init__(self, client: TogetherAIClient):
        self.client = client
    
    def create_sft_job(
        self,
        config: TrainingConfig,
        train_file_id: str,
        eval_file_id: str = None
    ) -> str:
        """Create an SFT fine-tuning job."""
        
        job_config = {
            "model": BASE_MODELS[config.base_model]["model_id"],
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
        return response.id
    
    def create_dpo_job(
        self,
        config: TrainingConfig,
        train_file_id: str,
        eval_file_id: str = None
    ) -> str:
        """Create a DPO fine-tuning job."""
        
        job_config = {
            "model": BASE_MODELS[config.base_model]["model_id"],
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
        return response.id
```

### 3.2. Job Monitoring

```python
import time
from datetime import datetime

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
            "created_at": response.created_at,
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
                "created_at": event.created_at,
                "level": event.level,
                "message": event.message,
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
            msg = event["message"]
            
            # Parse loss from message
            if "loss" in msg.lower():
                # Extract step and loss values
                import re
                
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
        callback = None
    ) -> dict:
        """Wait for job to complete."""
        
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.get_job_status(job_id)
            
            if callback:
                callback(status)
            
            if status["status"] == "succeeded":
                return status
            elif status["status"] == "failed":
                raise Exception(f"Training failed: {status.get('error')}")
            elif status["status"] == "cancelled":
                raise Exception("Training was cancelled")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Training timeout after {timeout}s")
```

### 3.3. Job Control

```python
class JobController:
    """Control training jobs."""
    
    def __init__(self, client: TogetherAIClient):
        self.client = client
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        
        try:
            self.client.client.fine_tuning.cancel(job_id)
            return True
        except Exception as e:
            print(f"Failed to cancel job: {e}")
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
                "created_at": job.created_at,
                "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            })
        
        return jobs
```

---

## 4. Training Pipeline

### 4.1. SFT Training Stage

```python
class SFTTrainingStage:
    """SFT training stage."""
    
    def __init__(
        self,
        client: TogetherAIClient,
        config: TrainingConfig
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
        output_dir: Path
    ) -> dict:
        """Run SFT training stage."""
        
        # Prepare data
        train_path = output_dir / "sft_train.jsonl"
        eval_path = output_dir / "sft_eval.jsonl"
        
        # Split data
        import random
        random.shuffle(sft_records)
        split_idx = int(len(sft_records) * 0.9)
        train_records = sft_records[:split_idx]
        eval_records = sft_records[split_idx:]
        
        # Prepare files
        self.preparer.prepare_sft_data(train_records, train_path)
        self.preparer.prepare_sft_data(eval_records, eval_path)
        
        # Validate
        valid_train, train_errors = self.validator.validate_sft_file(train_path)
        if not valid_train:
            raise Exception(f"Train data validation failed: {train_errors}")
        
        valid_eval, eval_errors = self.validator.validate_sft_file(eval_path)
        if not valid_eval:
            raise Exception(f"Eval data validation failed: {eval_errors}")
        
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
            eval_file_id
        )
        
        # Monitor
        result = self.monitor.wait_for_completion(
            job_id,
            callback=lambda s: print(f"SFT status: {s['status']}")
        )
        
        return {
            "stage": "sft",
            "job_id": job_id,
            "model_id": result["fine_tuned_model"],
            "train_records": len(train_records),
            "eval_records": len(eval_records),
        }
```

### 4.2. DPO Training Stage

```python
class DPOTrainingStage:
    """DPO training stage."""
    
    def __init__(
        self,
        client: TogetherAIClient,
        config: TrainingConfig
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
        output_dir: Path
    ) -> dict:
        """Run DPO training stage."""
        
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
        train_path = output_dir / "dpo_train.jsonl"
        eval_path = output_dir / "dpo_eval.jsonl"
        
        # Split data
        import random
        random.shuffle(dpo_records)
        split_idx = int(len(dpo_records) * 0.9)
        train_records = dpo_records[:split_idx]
        eval_records = dpo_records[split_idx:]
        
        # Prepare files
        self.preparer.prepare_dpo_data(train_records, train_path)
        self.preparer.prepare_dpo_data(eval_records, eval_path)
        
        # Validate
        valid_train, train_errors = self.validator.validate_dpo_file(train_path)
        if not valid_train:
            raise Exception(f"DPO train data validation failed: {train_errors}")
        
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
            eval_file_id
        )
        
        # Monitor
        result = self.monitor.wait_for_completion(
            job_id,
            callback=lambda s: print(f"DPO status: {s['status']}")
        )
        
        return {
            "stage": "dpo",
            "job_id": job_id,
            "model_id": result["fine_tuned_model"],
            "base_model": base_model_id,
            "train_records": len(train_records),
            "eval_records": len(eval_records),
        }
```

### 4.3. Complete Pipeline

```python
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

class V3TrainingPipeline:
    """Complete V3 training pipeline."""
    
    def __init__(
        self,
        together_api_key: str = None,
        config: TrainingConfig = None
    ):
        self.client = TogetherAIClient(together_api_key)
        self.config = config or TrainingConfig()
    
    async def run(
        self,
        sft_records: list,
        dpo_records: list,
        output_dir: Path
    ) -> PipelineResult:
        """Run complete training pipeline."""
        
        import time
        start_time = time.time()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: SFT
        sft_stage = SFTTrainingStage(self.client, self.config)
        sft_result = await sft_stage.run(sft_records, output_dir)
        
        print(f"SFT complete: {sft_result['model_id']}")
        
        # Stage 2: DPO
        dpo_stage = DPOTrainingStage(self.client, self.config)
        dpo_result = await dpo_stage.run(
            dpo_records,
            sft_result["model_id"],
            output_dir
        )
        
        print(f"DPO complete: {dpo_result['model_id']}")
        
        end_time = time.time()
        
        return PipelineResult(
            sft_model_id=sft_result["model_id"],
            dpo_model_id=dpo_result["model_id"],
            sft_job_id=sft_result["job_id"],
            dpo_job_id=dpo_result["job_id"],
            total_sft_records=sft_result["train_records"] + sft_result["eval_records"],
            total_dpo_records=dpo_result["train_records"] + dpo_result["eval_records"],
            training_time_seconds=end_time - start_time,
        )
```

---

## 5. Checkpoint Evaluation

### 5.1. Model Inference

```python
class ModelInference:
    """Inference with trained model."""
    
    def __init__(self, client: TogetherAIClient, model_id: str):
        self.client = client
        self.model_id = model_id
    
    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.3
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
        max_tokens: int = 2048
    ) -> list[str]:
        """Batch generate responses."""
        
        responses = []
        for messages in prompts:
            response = self.generate(messages, max_tokens)
            responses.append(response)
        
        return responses
```

### 5.2. Regression Testing

```python
class RegressionTester:
    """Run regression tests on trained model."""
    
    def __init__(self, inference: ModelInference):
        self.inference = inference
    
    def run_eval_cases(
        self,
        eval_cases: list[dict]
    ) -> dict:
        """Run evaluation cases."""
        
        results = {
            "total": len(eval_cases),
            "passed": 0,
            "failed": 0,
            "failures": [],
        }
        
        for case in eval_cases:
            # Build messages
            messages = case["input"]["messages"]
            
            # Generate response
            response = self.inference.generate(messages)
            
            # Check constraints
            passed, failures = self._check_constraints(
                response,
                case.get("checks", {})
            )
            
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
        
        results["pass_rate"] = results["passed"] / results["total"]
        
        return results
    
    def _check_constraints(
        self,
        response: str,
        checks: dict
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
                import json
                # Extract JSON from code block
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
                if json_match:
                    json.loads(json_match.group(1))
                else:
                    # Try parsing entire response
                    json.loads(response)
            except:
                failures.append("Not valid JSON format")
        
        return len(failures) == 0, failures
```

### 5.3. A/B Comparison

```python
class ABComparison:
    """Compare V3 model against baseline."""
    
    def __init__(
        self,
        v3_model: ModelInference,
        baseline_model: ModelInference
    ):
        self.v3 = v3_model
        self.baseline = baseline_model
    
    def compare_on_prompts(
        self,
        prompts: list[list[dict]]
    ) -> dict:
        """Compare models on prompts."""
        
        results = {
            "v3_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
            "comparisons": [],
        }
        
        for messages in prompts:
            v3_response = self.v3.generate(messages)
            baseline_response = self.baseline.generate(messages)
            
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
        results["v3_win_rate"] = results["v3_wins"] / total
        results["baseline_win_rate"] = results["baseline_wins"] / total
        
        return results
    
    def _score_response(
        self,
        response: str,
        messages: list[dict]
    ) -> float:
        """Score a response (higher is better)."""
        
        score = 0.0
        
        # Penalize permission-seeking
        permission_phrases = [
            "would you like me to",
            "should i",
            "do you want me to",
            "can i proceed",
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
```

---

## 6. CLI Interface

```python
import click
from pathlib import Path

@click.group()
def cli():
    """CognitiveTwin V3 Training Pipeline."""
    pass

@cli.command()
@click.option("--sft-data", type=Path, required=True)
@click.option("--dpo-data", type=Path, required=True)
@click.option("--output-dir", type=Path, required=True)
@click.option("--base-model", default="llama-3.1-8b")
@click.option("--epochs", default=3)
def train(sft_data, dpo_data, output_dir, base_model, epochs):
    """Run training pipeline."""
    
    import asyncio
    import json
    
    # Load data
    with open(sft_data) as f:
        sft_records = [json.loads(line) for line in f]
    
    with open(dpo_data) as f:
        dpo_records = [json.loads(line) for line in f]
    
    # Configure
    config = TrainingConfig(
        base_model=base_model,
        num_epochs=epochs,
    )
    
    # Run
    pipeline = V3TrainingPipeline(config=config)
    result = asyncio.run(pipeline.run(sft_records, dpo_records, output_dir))
    
    click.echo(f"Training complete!")
    click.echo(f"SFT Model: {result.sft_model_id}")
    click.echo(f"DPO Model: {result.dpo_model_id}")

@cli.command()
@click.option("--model-id", required=True)
@click.option("--eval-data", type=Path, required=True)
def evaluate(model_id, eval_data):
    """Run evaluation on trained model."""
    
    import json
    
    # Load eval cases
    with open(eval_data) as f:
        eval_cases = [json.loads(line) for line in f]
    
    # Run evaluation
    client = TogetherAIClient()
    inference = ModelInference(client, model_id)
    tester = RegressionTester(inference)
    
    results = tester.run_eval_cases(eval_cases)
    
    click.echo(f"Evaluation Results:")
    click.echo(f"  Total: {results['total']}")
    click.echo(f"  Passed: {results['passed']}")
    click.echo(f"  Failed: {results['failed']}")
    click.echo(f"  Pass Rate: {results['pass_rate']:.2%}")

if __name__ == "__main__":
    cli()
```

