#!/usr/bin/env python3
"""
Batch Evaluation Script for Multi-Model Testing Framework

This script runs multiple evaluation experiments with different model configurations,
useful for comparing different models, tasks, and evaluation modes.
"""

import subprocess
import sys
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time

class BatchEvaluator:
    """Run multiple evaluation experiments"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.experiments = self.load_experiments()
        self.results = []
        
    def load_experiments(self) -> List[Dict[str, Any]]:
        """Load experiment configurations from JSON file"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('experiments', [])
            
    def run_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment"""
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment.get('name', 'unnamed')}")
        print(f"{'='*60}")
        
        # Determine which test script to use based on model
        model_name = experiment.get('model', '')
        test_script = self.get_test_script(model_name)
        
        if not test_script:
            return {
                "name": experiment.get("name", "unnamed"),
                "config": experiment,
                "success": False,
                "error": f"No test script found for model: {model_name}",
                "duration": 0
            }
        
        # Build command
        cmd = [sys.executable, test_script]
        
        # Add arguments
        for key, value in experiment.items():
            if key in ["name", "model"]:
                continue
                
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))
                
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            success = True
            error_msg = None
        except subprocess.CalledProcessError as e:
            success = False
            error_msg = e.stderr
            result = e
            
        end_time = time.time()
        
        experiment_result = {
            "name": experiment.get("name", "unnamed"),
            "model": model_name,
            "test_script": test_script,
            "config": experiment,
            "success": success,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "error": error_msg,
            "stdout": result.stdout if success else None,
            "stderr": result.stderr if success else None
        }
        
        if success:
            print(f"✓ Experiment completed successfully in {experiment_result['duration']:.2f}s")
        else:
            print(f"✗ Experiment failed: {error_msg}")
            
        return experiment_result
        
    def get_test_script(self, model_name: str) -> str:
        """Get the appropriate test script for a model"""
        script_mapping = {
            'qwen2-vl-2b-awq': 'eval/eval_qwen2_vl_2b_awq.py',
            'qwen25-omni-3b-gguf': 'eval/eval_qwen25_omni_3b_gguf.py',
            'qwen25-vl-3b': 'eval/eval_qwen25_vl_3b.py',
            'qwen25-vl-7b': 'eval/eval_qwen25_vl_7b.py',
            'qwen2-vl-7b': 'eval/eval_qwen2_vl_7b.py',
            'llava-7b': 'eval/eval_llava_7b.py',
            'llava-next-7b': 'eval/eval_llava_next_7b.py',
            'llava-onevision-7b': 'eval/eval_llava_onevision_7b.py',
            'llava-onevision-chat-7b': 'eval/eval_llava_onevision_chat_7b.py',
            'llava-next-interleave-7b': 'eval/eval_llava_next_interleave_7b.py',
            'llama-3-8b': 'eval/eval_llama_3_8b.py',
            'llama-32-11b': 'eval/eval_llama_32_11b_11b.py',
            'internvl3-9b': 'eval/eval_internvl3_9b_9b.py',
            'qwen3-8b': 'eval/eval_qwen3_8b_8b.py'
        }
        
        return script_mapping.get(model_name, '')
        
    def run_all_experiments(self, filter_model: str = None) -> List[Dict[str, Any]]:
        """Run all experiments in the configuration"""
        # Filter experiments by model if specified
        experiments_to_run = self.experiments
        if filter_model:
            experiments_to_run = [exp for exp in self.experiments if exp.get('model') == filter_model]
            print(f"Filtering experiments for model: {filter_model}")
            print(f"Found {len(experiments_to_run)} experiments for this model")
        
        print(f"Starting batch evaluation with {len(experiments_to_run)} experiments")
        print(f"Configuration file: {self.config_file}")
        
        results = []
        for i, experiment in enumerate(experiments_to_run, 1):
            print(f"\n[{i}/{len(experiments_to_run)}]")
            result = self.run_single_experiment(experiment)
            results.append(result)
            
        return results
        
    def save_batch_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save batch evaluation results"""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        batch_report = {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(results),
            "successful_experiments": sum(1 for r in results if r["success"]),
            "failed_experiments": sum(1 for r in results if not r["success"]),
            "total_duration": sum(r["duration"] for r in results),
            "experiments": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)
            
        print(f"\nBatch results saved to: {output_file}")
        
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print batch evaluation summary"""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total Duration: {sum(r['duration'] for r in results):.2f}s")
        print(f"Average Duration: {sum(r['duration'] for r in results) / len(results):.2f}s")
        
        if successful:
            print("\nSuccessful Experiments:")
            for result in successful:
                print(f"  ✓ {result['name']} ({result['model']}) - {result['duration']:.2f}s")
                
        if failed:
            print("\nFailed Experiments:")
            for result in failed:
                print(f"  ✗ {result['name']} ({result['model']}) - {result['error']}")
                
        print("="*60)

def create_example_config():
    """Create an example configuration file"""
    example_config = {
        "experiments": [
            {
                "name": "qwen2-vl-2b-awq_understanding",
                "model": "qwen2-vl-2b-awq",
                "data-path": "../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl",
                "task": "understanding",
                "num-samples": "10",
                "output-dir": "results/batch_qwen2_vl_2b_awq_understanding"
            },
            {
                "name": "qwen25-vl-3b_understanding",
                "model": "qwen25-vl-3b",
                "data-path": "../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl",
                "task": "understanding",
                "num-samples": "10",
                "output-dir": "results/batch_qwen25_vl_3b_understanding"
            },
            {
                "name": "qwen2-vl-2b-awq_reasoning",
                "model": "qwen2-vl-2b-awq",
                "data-path": "../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl",
                "task": "reasoning",
                "num-samples": "10",
                "output-dir": "results/batch_qwen2_vl_2b_awq_reasoning"
            },
            {
                "name": "qwen25-vl-3b_reasoning",
                "model": "qwen25-vl-3b",
                "data-path": "../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl",
                "task": "reasoning",
                "num-samples": "10",
                "output-dir": "results/batch_qwen25_vl_3b_reasoning"
            },
            {
                "name": "qwen2-vl-2b-awq_locating",
                "model": "qwen2-vl-2b-awq",
                "data-path": "../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl",
                "task": "locating",
                "num-samples": "10",
                "output-dir": "results/batch_qwen2_vl_2b_awq_locating"
            },
            {
                "name": "qwen25-vl-3b_locating",
                "model": "qwen25-vl-3b",
                "data-path": "../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl",
                "task": "locating",
                "num-samples": "10",
                "output-dir": "results/batch_qwen25_vl_3b_locating"
            }
        ]
    }
    
    config_file = "configs/batch_config_example.json"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
        
    print(f"Example configuration file created: {config_file}")
    return config_file

def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for multi-model experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_eval.py --config configs/batch_config.json
  
  python batch_eval.py --config configs/batch_config.json --output batch_results.json
  
  python batch_eval.py --config configs/batch_config.json --model qwen2-vl-2b-awq
  
  python batch_eval.py --create-example
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to batch configuration JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/batch_evaluation_results.json",
        help="Output file for batch results (default: results/batch_evaluation_results.json)"
    )
    
    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create an example configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Filter experiments to run only for the specified model"
    )
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config()
        return
        
    if not args.config:
        print("Error: Please specify a configuration file with --config")
        print("Use --create-example to create an example configuration file")
        return
        
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
        
    # Run batch evaluation
    evaluator = BatchEvaluator(args.config)
    results = evaluator.run_all_experiments(filter_model=args.model)
    
    # Save results
    evaluator.save_batch_results(results, args.output)
    
    # Print summary
    evaluator.print_summary(results)

if __name__ == "__main__":
    main()
