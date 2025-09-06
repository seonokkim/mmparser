#!/usr/bin/env python3
"""
Metrics Calculator for Multi-Model Testing Framework

This module provides comprehensive metrics calculation capabilities for model evaluation results,
following top-tier research paper standards.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from pathlib import Path

class MetricsCalculator:
    """Calculate various metrics from evaluation results"""
    
    def __init__(self, results_file: Optional[str] = None, results_data: Optional[Dict[str, Any]] = None):
        if results_file:
            self.results = self.load_results(results_file)
        elif results_data:
            self.results = results_data
        else:
            raise ValueError("Either results_file or results_data must be provided")
            
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load evaluation results from JSON file"""
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic accuracy and F1 metrics"""
        detailed_results = self.results.get('detailed_results', [])
        
        if not detailed_results:
            return {"accuracy": 0.0, "f1_score": 0.0}
            
        total_samples = len(detailed_results)
        correct_samples = sum(1 for r in detailed_results if r.get('is_correct', False))
        
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
        f1_score = self.calculate_f1_score(detailed_results)
        
        return {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "total_samples": total_samples,
            "correct_samples": correct_samples
        }
        
    def calculate_f1_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate F1 score for evaluation results"""
        # Simplified F1 calculation - in practice, you'd need more sophisticated text matching
        correct = 0
        total = len(results)
        
        for result in results:
            predicted = result.get('model_response', '').strip().lower()
            ground_truth = result.get('ground_truth', '').strip().lower()
            
            # Simple exact match for now
            if predicted == ground_truth:
                correct += 1
                
        precision = correct / total if total > 0 else 0.0
        recall = correct / total if total > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
        
    def calculate_task_specific_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each task type"""
        detailed_results = self.results.get('detailed_results', [])
        
        task_metrics = defaultdict(list)
        
        for result in detailed_results:
            task = result.get('task', 'unknown').lower()
            is_correct = result.get('is_correct', False)
            task_metrics[task].append(is_correct)
            
        # Calculate metrics for each task
        task_results = {}
        for task, correct_list in task_metrics.items():
            if correct_list:
                accuracy = sum(correct_list) / len(correct_list)
                task_results[task] = {
                    "accuracy": accuracy,
                    "total_samples": len(correct_list),
                    "correct_samples": sum(correct_list)
                }
                
        return task_results
        
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance and timing metrics"""
        detailed_results = self.results.get('detailed_results', [])
        
        if not detailed_results:
            return {}
            
        evaluation_times = [r.get('evaluation_time', 0) for r in detailed_results]
        
        return {
            "total_evaluation_time": sum(evaluation_times),
            "avg_evaluation_time": np.mean(evaluation_times),
            "min_evaluation_time": np.min(evaluation_times),
            "max_evaluation_time": np.max(evaluation_times),
            "std_evaluation_time": np.std(evaluation_times)
        }
        
    def calculate_model_specific_metrics(self) -> Dict[str, Any]:
        """Calculate model-specific metrics"""
        detailed_results = self.results.get('detailed_results', [])
        
        # Analyze response quality
        response_lengths = []
        response_quality_scores = []
        
        for result in detailed_results:
            model_response = result.get('model_response', '')
            response_lengths.append(len(model_response))
            
            # Simple quality score based on response length and content
            quality_score = min(1.0, len(model_response) / 100)  # Normalize by expected length
            response_quality_scores.append(quality_score)
        
        return {
            "avg_response_length": np.mean(response_lengths) if response_lengths else 0,
            "avg_response_quality": np.mean(response_quality_scores) if response_quality_scores else 0,
            "total_responses": len(detailed_results)
        }
        
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        basic_metrics = self.calculate_basic_metrics()
        task_metrics = self.calculate_task_specific_metrics()
        performance_metrics = self.calculate_performance_metrics()
        model_metrics = self.calculate_model_specific_metrics()
        
        return {
            "basic_metrics": basic_metrics,
            "task_specific_metrics": task_metrics,
            "performance_metrics": performance_metrics,
            "model_specific_metrics": model_metrics
        }
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive metrics report"""
        comprehensive_metrics = self.calculate_comprehensive_metrics()
        
        # Get experiment metadata
        experiment_id = self.results.get('experiment_id', 'unknown')
        timestamp = self.results.get('timestamp', 'unknown')
        model_name = self.results.get('model_name', 'unknown')
        
        report = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "metrics": comprehensive_metrics,
            "calculation_timestamp": self.get_current_timestamp()
        }
        
        return report
        
    def save_report(self, output_file: str):
        """Save metrics report to file"""
        report = self.generate_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"Metrics report saved to: {output_file}")
        
    def print_summary(self):
        """Print a summary of the metrics"""
        basic_metrics = self.calculate_basic_metrics()
        task_metrics = self.calculate_task_specific_metrics()
        performance_metrics = self.calculate_performance_metrics()
        model_metrics = self.calculate_model_specific_metrics()
        
        print("=" * 60)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 60)
        
        print(f"Experiment ID: {self.results.get('experiment_id', 'unknown')}")
        print(f"Model: {self.results.get('model_name', 'unknown')}")
        print(f"Timestamp: {self.results.get('timestamp', 'unknown')}")
        print()
        
        print("BASIC METRICS:")
        print(f"  Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {basic_metrics['f1_score']:.4f}")
        print(f"  Total Samples: {basic_metrics['total_samples']}")
        print(f"  Correct Samples: {basic_metrics['correct_samples']}")
        print()
        
        print("TASK-SPECIFIC METRICS:")
        for task, metrics in task_metrics.items():
            print(f"  {task.title()}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Samples: {metrics['total_samples']}")
        print()
        
        print("MODEL-SPECIFIC METRICS:")
        print(f"  Avg Response Length: {model_metrics['avg_response_length']:.1f} chars")
        print(f"  Avg Response Quality: {model_metrics['avg_response_quality']:.4f}")
        print(f"  Total Responses: {model_metrics['total_responses']}")
        print()
        
        print("PERFORMANCE METRICS:")
        if performance_metrics:
            print(f"  Total Time: {performance_metrics['total_evaluation_time']:.2f}s")
            print(f"  Avg Time per Sample: {performance_metrics['avg_evaluation_time']:.2f}s")
            print(f"  Min Time: {performance_metrics['min_evaluation_time']:.2f}s")
            print(f"  Max Time: {performance_metrics['max_evaluation_time']:.2f}s")
        print("=" * 60)
        
    def get_current_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """Command line interface for metrics calculation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate metrics from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics.py --results-file results/evaluation_detailed.json
  
  python metrics.py --results-file results/evaluation_detailed.json --output-file metrics_report.json
        """
    )
    
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to evaluation results JSON file"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for metrics report (optional)"
    )
    
    parser.add_argument(
        "--print-summary",
        action="store_true",
        default=True,
        help="Print metrics summary to console (default: True)"
    )
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return
        
    # Calculate metrics
    calculator = MetricsCalculator(args.results_file)
    
    # Print summary
    if args.print_summary:
        calculator.print_summary()
        
    # Save report if output file specified
    if args.output_file:
        calculator.save_report(args.output_file)

if __name__ == "__main__":
    main()

