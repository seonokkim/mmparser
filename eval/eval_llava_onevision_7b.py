#!/usr/bin/env python3
"""
Test script for LLaVA-OneVision-7B model

This script provides comprehensive testing capabilities for the LLaVA-OneVision-7B model,
following top-tier research paper standards for model evaluation.
"""

import os
import sys
import json
import argparse
import time
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.model_config import get_model_config, get_model_info
from utils.evaluator import BaseEvaluator
from utils.metrics import MetricsCalculator

# Import model classes
try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

@dataclass
class LLaVAOneVision7BConfig:
    """Configuration for LLaVA-OneVision-7B evaluation"""
    # Model settings
    model_name: str = "llava-onevision-7b"
    data_path: str = ""
    task: str = "understanding"
    
    # Evaluation settings
    num_samples: int = -1  # -1 means all samples
    output_dir: str = "results/llava-onevision-7b"
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    
    # Device settings
    device: str = "auto"
    batch_size: int = 1
    
    # Processing settings
    save_detailed_results: bool = True
    use_gpt_extraction: bool = False
    free_form: bool = True

class LLaVAOneVision7BEvaluator(BaseEvaluator):
    """Evaluator for LLaVA-OneVision-7B model"""
    
    def __init__(self, config: LLaVAOneVision7BConfig):
        super().__init__(config)
        self.config = config
        self.model_config = get_model_config(config.model_name)
        self.model = None
        self.processor = None
        
        # Setup logging
        self.setup_logging()
        
        # Validate model availability
        self.validate_setup()
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_dir}/llava_onevision_7b_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_setup(self):
        """Validate that all required components are available"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers")
            
        if not PIL_AVAILABLE:
            raise ImportError("PIL library not available. Install with: pip install Pillow")
            
        # Check model path
        model_path = Path(self.model_config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        # Check for model files
        has_safetensors = any(model_path.glob("*.safetensors"))
        if not has_safetensors:
            raise FileNotFoundError(f"No safetensors files found in: {model_path}")
            
        self.logger.info(f"Model validation passed for {self.config.model_name}")
        
    def load_model(self):
        """Load LLaVA-OneVision-7B model and processor"""
        self.logger.info(f"Loading {self.config.model_name} model...")
        
        try:
            # Load model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_config.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if self.config.device == "auto" else self.config.device,
                low_cpu_mem_usage=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_config.model_path)
            
            self.logger.info(f"Successfully loaded {self.config.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def generate_response(self, question: str, image_paths: List[str]) -> str:
        """Generate response using LLaVA-OneVision-7B model"""
        try:
            # Load and process images
            images = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    images.append(image)
                    
            if not images:
                return "No valid images found."
            
            # Prepare conversation for LLaVA
            conversation = [
                {
                    "role": "user",
                    "content": f"<image>\n{question}"
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(text=text, images=images[0], return_tensors="pt")
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True
                )
                
                full_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Extract only the assistant's response
                if "assistant" in full_response:
                    parts = full_response.split("assistant")
                    if len(parts) > 1:
                        response = parts[-1].strip()
                    else:
                        response = full_response
                else:
                    response = full_response
                    
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
            
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample"""
        start_time = time.time()
        
        # Extract sample information
        question = sample.get("question", "")
        images = sample.get("images", [])
        ground_truth = sample.get("answer", "")
        question_id = sample.get("question_id", "")
        
        # Generate response
        response = self.generate_response(question, images)
        
        # Calculate accuracy (simplified)
        is_correct = self.calculate_accuracy(response, ground_truth)
        
        evaluation_time = time.time() - start_time
        
        return {
            "question_id": question_id,
            "question": question,
            "images": images,
            "ground_truth": ground_truth,
            "model_response": response,
            "is_correct": is_correct,
            "evaluation_time": evaluation_time,
            "model_name": self.config.model_name
        }
        
    def calculate_accuracy(self, predicted: str, ground_truth: Any) -> bool:
        """Calculate if prediction matches ground truth"""
        predicted_str = str(predicted).strip().lower()
        ground_truth_str = str(ground_truth).strip().lower()
        return predicted_str == ground_truth_str
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation"""
        self.logger.info(f"Starting {self.config.model_name} evaluation...")
        
        # Load model
        self.load_model()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Run evaluation
        results = []
        for i, sample in enumerate(dataset):
            self.logger.info(f"Processing sample {i+1}/{len(dataset)}")
            result = self.evaluate_sample(sample)
            results.append(result)
            
        # Calculate metrics
        total_correct = sum(1 for r in results if r["is_correct"])
        accuracy = total_correct / len(results) if results else 0.0
        
        # Create experiment result
        experiment_result = {
            "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.config.model_name}_{self.config.task}",
            "timestamp": datetime.now().isoformat(),
            "model_name": self.config.model_name,
            "model_info": get_model_info(self.config.model_name),
            "config": asdict(self.config),
            "dataset_path": self.config.data_path,
            "task": self.config.task,
            "total_samples": len(dataset),
            "processed_samples": len(results),
            "metrics": {
                "accuracy": accuracy,
                "total_correct": total_correct,
                "total_samples": len(results)
            },
            "detailed_results": results,
            "total_time": sum(r["evaluation_time"] for r in results),
            "avg_time_per_sample": sum(r["evaluation_time"] for r in results) / len(results) if results else 0.0
        }
        
        # Save results
        self.save_results(experiment_result)
        
        self.logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return experiment_result
        
    def save_results(self, result: Dict[str, Any]):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.config.model_name}_{self.config.task}"
        
        # Save detailed results
        if self.config.save_detailed_results:
            detailed_path = os.path.join(self.config.output_dir, f"{filename}_detailed.json")
            with open(detailed_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Detailed results saved to: {detailed_path}")
            
        # Save summary
        summary_path = os.path.join(self.config.output_dir, f"{filename}_summary.json")
        summary = {
            "experiment_id": result["experiment_id"],
            "timestamp": result["timestamp"],
            "model_name": result["model_name"],
            "config": result["config"],
            "metrics": result["metrics"],
            "performance": {
                "total_time": result["total_time"],
                "avg_time_per_sample": result["avg_time_per_sample"]
            }
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description="LLaVA-OneVision-7B Model Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python test_llava_onevision_7b.py --data-path /path/to/data.jsonl --task understanding --num-samples 100
  
  # Full evaluation with custom output
  python test_llava_onevision_7b.py --data-path /path/to/data.jsonl --task understanding --num-samples 100 --output-dir results/custom
        """
    )
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to dataset (JSONL file or directory)")
    parser.add_argument("--task", type=str, default="understanding",
                       help="Task to evaluate (understanding, reasoning, locating)")
    parser.add_argument("--num-samples", type=int, default=-1,
                       help="Number of samples to evaluate (-1 for all)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results/llava-onevision-7b",
                       help="Output directory for results")
    
    # Generation arguments
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Temperature for generation")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for inference")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create configuration
    config = LLaVAOneVision7BConfig(
        data_path=args.data_path,
        task=args.task,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    evaluator = LLaVAOneVision7BEvaluator(config)
    result = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "="*50)
    print("LLAVA-ONEVISION-7B EVALUATION SUMMARY")
    print("="*50)
    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Model: {result['model_name']}")
    print(f"Task: {result['task']}")
    print(f"Total Samples: {result['total_samples']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Total Time: {result['total_time']:.2f}s")
    print(f"Avg Time per Sample: {result['avg_time_per_sample']:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()


