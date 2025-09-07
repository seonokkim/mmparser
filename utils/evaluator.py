#!/usr/bin/env python3
"""
Base Evaluator Class for Multi-Model Testing Framework

This module provides the base evaluator class that all model-specific evaluators inherit from,
following top-tier research paper standards for model evaluation.
"""

import os
import json
import logging
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .data_loader import LongDocURLDataLoader

# Import Azure OpenAI for answer extraction
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

@dataclass
class BaseConfig:
    """Base configuration class for all evaluators"""
    data_path: str
    task: str
    num_samples: int = -1
    output_dir: str = "results"
    save_detailed_results: bool = True

class BaseEvaluator(ABC):
    """Base evaluator class for all model evaluations"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = None
        
    def setup_logging(self, log_file: str = "evaluation.log"):
        """Setup logging configuration"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, log_file)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from the specified path"""
        self.logger.info(f"Loading dataset from: {self.config.data_path}")
        
        # Use LongDocURL data loader
        data_loader = LongDocURLDataLoader("/workspace/data/LongDocURL")
        
        # Determine dataset file
        if self.config.data_path.endswith('.jsonl'):
            dataset_file = os.path.basename(self.config.data_path)
        else:
            # Default to 10% dataset
            dataset_file = "LongDocURL_public_with_subtask_category_10pct.jsonl"
            
        # Load dataset
        dataset = data_loader.load_dataset(dataset_file)
        
        # Filter by task if specified
        if self.config.task != "all":
            dataset = data_loader.filter_by_task(dataset, self.config.task)
            
        # Limit number of samples
        if self.config.num_samples > 0:
            dataset = data_loader.filter_by_num_samples(dataset, self.config.num_samples)
            
        self.logger.info(f"Loaded {len(dataset)} samples")
        return dataset
        
    def load_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        dataset = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                if "question_id" not in data:
                    data["question_id"] = i
                dataset.append(data)
        return dataset
        
    def calculate_accuracy(self, predicted: str, ground_truth: Any) -> bool:
        """Calculate if prediction matches ground truth (basic implementation)"""
        predicted_str = str(predicted).strip().lower()
        ground_truth_str = str(ground_truth).strip().lower()
        return predicted_str == ground_truth_str
        
    def save_results(self, result: Dict[str, Any], filename_prefix: str = "evaluation"):
        """Save evaluation results"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename_prefix}"
        
        # Save detailed results
        if self.config.save_detailed_results:
            detailed_path = os.path.join(self.config.output_dir, f"{filename}_detailed.json")
            with open(detailed_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Detailed results saved to: {detailed_path}")
            
        # Save summary
        summary_path = os.path.join(self.config.output_dir, f"{filename}_summary.json")
        summary = {
            "experiment_id": result.get("experiment_id", "unknown"),
            "timestamp": result.get("timestamp", "unknown"),
            "config": result.get("config", {}),
            "metrics": result.get("metrics", {}),
            "performance": result.get("performance", {})
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Summary saved to: {summary_path}")
        
    @abstractmethod
    def load_model(self):
        """Load the model (to be implemented by subclasses)"""
        pass
        
    @abstractmethod
    def generate_response(self, question: str, image_paths: List[str]) -> str:
        """Generate response using the model (to be implemented by subclasses)"""
        pass
        
    @abstractmethod
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample (to be implemented by subclasses)"""
        pass
        
    @abstractmethod
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation (to be implemented by subclasses)"""
        pass
