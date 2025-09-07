#!/usr/bin/env python3
"""
Simple test script to check if data loader works
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.data_loader import LongDocURLDataLoader

def main():
    print("Testing LongDocURL Data Loader...")
    
    # Initialize data loader
    data_loader = LongDocURLDataLoader("/workspace/data/LongDocURL")
    
    # Load dataset
    try:
        dataset = data_loader.load_dataset("LongDocURL_public_with_subtask_category_10pct.jsonl")
        print(f"Successfully loaded {len(dataset)} samples")
        
        # Show first sample
        if dataset:
            sample = dataset[0]
            print("\nFirst sample:")
            print(f"  Question ID: {sample['question_id']}")
            print(f"  Question: {sample['question'][:100]}...")
            print(f"  Task: {sample['task']}")
            print(f"  Answer: {sample['answer']}")
            print(f"  Images: {len(sample['images'])} images")
            if sample['images']:
                print(f"  First image path: {sample['images'][0]}")
                print(f"  Image exists: {os.path.exists(sample['images'][0])}")
        
        # Filter by task
        reasoning_samples = data_loader.filter_by_task(dataset, "reasoning")
        print(f"\nReasoning samples: {len(reasoning_samples)}")
        
        # Limit to 1 sample
        one_sample = data_loader.filter_by_num_samples(reasoning_samples, 1)
        print(f"Limited to 1 sample: {len(one_sample)}")
        
        if one_sample:
            sample = one_sample[0]
            print(f"\nTest sample:")
            print(f"  Question: {sample['question']}")
            print(f"  Answer: {sample['answer']}")
            print(f"  Images: {sample['images']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
