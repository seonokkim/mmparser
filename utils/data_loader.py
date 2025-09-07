#!/usr/bin/env python3
"""
Data Loader for LongDocURL Dataset

This module provides utilities for loading and processing LongDocURL dataset,
following the format used in the original LongDocURL paper.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class LongDocURLSample:
    """LongDocURL sample data structure"""
    question_id: str
    doc_no: str
    total_pages: int
    start_end_idx: List[int]
    question_type: str
    question: str
    answer: Any
    detailed_evidences: str
    evidence_pages: List[int]
    evidence_sources: List[str]
    answer_format: str
    task_tag: str
    images: List[str]
    pdf_path: str
    subTask: List[str]

class LongDocURLDataLoader:
    """Data loader for LongDocURL dataset"""
    
    def __init__(self, data_base_dir: str):
        self.data_base_dir = Path(data_base_dir)
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self, dataset_file: str = "LongDocURL_public_with_subtask_category_10pct.jsonl") -> List[Dict[str, Any]]:
        """Load LongDocURL dataset from JSONL file"""
        dataset_path = self.data_base_dir / dataset_file
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        self.logger.info(f"Loading LongDocURL dataset from: {dataset_path}")
        
        dataset = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Convert LongDocURL format to our standard format
                    converted_sample = self.convert_longdocurl_format(data)
                    dataset.append(converted_sample)
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error processing line {line_num}: {e}")
                    continue
                    
        self.logger.info(f"Loaded {len(dataset)} samples from LongDocURL dataset")
        return dataset
        
    def convert_longdocurl_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LongDocURL format to our standard format"""
        # Convert image paths to local paths
        local_images = []
        for img_path in sample.get("images", []):
            local_path = self.convert_image_path(img_path)
            if local_path:
                local_images.append(local_path)
                
        # Map task_tag to our task format
        task_mapping = {
            "Understanding": "understanding",
            "Reasoning": "reasoning", 
            "Locating": "locating"
        }
        
        task = task_mapping.get(sample.get("task_tag", ""), "understanding")
        
        return {
            "question_id": sample.get("question_id", ""),
            "question": sample.get("question", ""),
            "images": local_images,
            "answer": sample.get("answer", ""),
            "task": task,
            # Additional LongDocURL specific fields
            "doc_no": sample.get("doc_no", ""),
            "total_pages": sample.get("total_pages", 0),
            "start_end_idx": sample.get("start_end_idx", []),
            "question_type": sample.get("question_type", ""),
            "detailed_evidences": sample.get("detailed_evidences", ""),
            "evidence_pages": sample.get("evidence_pages", []),
            "evidence_sources": sample.get("evidence_sources", []),
            "answer_format": sample.get("answer_format", "String"),
            "task_tag": sample.get("task_tag", ""),
            "pdf_path": sample.get("pdf_path", ""),
            "subTask": sample.get("subTask", [])
        }
        
    def convert_image_path(self, original_path: str) -> Optional[str]:
        """Convert original image path to local path"""
        # Original path format: /data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4045/4045421_4.png
        # Local path format: ../data/LongDocURL/pdf_pngs/4045/4045421_4.png
        
        try:
            # Extract the relevant parts from the original path
            path_parts = Path(original_path).parts
            
            # Find the pdf_pngs directory index
            pdf_pngs_idx = None
            for i, part in enumerate(path_parts):
                if part == "pdf_pngs":
                    pdf_pngs_idx = i
                    break
                    
            if pdf_pngs_idx is None:
                self.logger.warning(f"Could not find pdf_pngs in path: {original_path}")
                return None
                
            # Get the subdirectory and filename
            subdir = path_parts[pdf_pngs_idx + 2]  # Skip pdf_pngs and 4000-4999
            filename = path_parts[-1]
            
            # Construct local path
            local_path = self.data_base_dir / "pdf_pngs" / subdir / filename
            
            # Check if file exists
            if local_path.exists():
                return str(local_path)
            else:
                self.logger.warning(f"Image file not found: {local_path}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error converting image path {original_path}: {e}")
            return None
            
    def filter_by_task(self, dataset: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
        """Filter dataset by task type"""
        if task == "all":
            return dataset
            
        filtered = [sample for sample in dataset if sample.get("task", "").lower() == task.lower()]
        self.logger.info(f"Filtered {len(filtered)} samples for task: {task}")
        return filtered
        
    def filter_by_num_samples(self, dataset: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        """Limit dataset to specified number of samples"""
        if num_samples <= 0:
            return dataset
            
        limited = dataset[:num_samples]
        self.logger.info(f"Limited dataset to {len(limited)} samples")
        return limited
        
    def get_dataset_info(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get information about the dataset"""
        if not dataset:
            return {}
            
        task_counts = {}
        answer_format_counts = {}
        evidence_source_counts = {}
        
        for sample in dataset:
            # Count by task
            task = sample.get("task", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
            
            # Count by answer format
            answer_format = sample.get("answer_format", "unknown")
            answer_format_counts[answer_format] = answer_format_counts.get(answer_format, 0) + 1
            
            # Count by evidence sources
            evidence_sources = sample.get("evidence_sources", [])
            for source in evidence_sources:
                evidence_source_counts[source] = evidence_source_counts.get(source, 0) + 1
                
        return {
            "total_samples": len(dataset),
            "task_distribution": task_counts,
            "answer_format_distribution": answer_format_counts,
            "evidence_source_distribution": evidence_source_counts
        }

def main():
    """Test the data loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LongDocURL data loader")
    parser.add_argument("--data-base-dir", type=str, default="../data/LongDocURL",
                       help="Base directory for LongDocURL data")
    parser.add_argument("--dataset-file", type=str, 
                       default="LongDocURL_public_with_subtask_category_10pct.jsonl",
                       help="Dataset file to load")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to display")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    loader = LongDocURLDataLoader(args.data_base_dir)
    dataset = loader.load_dataset(args.dataset_file)
    
    # Get dataset info
    info = loader.get_dataset_info(dataset)
    print("Dataset Information:")
    print(json.dumps(info, indent=2))
    
    # Display sample samples
    print(f"\nFirst {args.num_samples} samples:")
    for i, sample in enumerate(dataset[:args.num_samples]):
        print(f"\nSample {i+1}:")
        print(f"  Question ID: {sample['question_id']}")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Task: {sample['task']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Images: {len(sample['images'])} images")
        print(f"  Evidence Pages: {sample['evidence_pages']}")

if __name__ == "__main__":
    main()


