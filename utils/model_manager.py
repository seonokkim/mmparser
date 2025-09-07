#!/usr/bin/env python3
"""
Model Manager for Multi-Model Testing Framework

This module provides utilities for downloading, validating, and managing models,
including automatic re-download of corrupted models.
"""

import os
import sys
import json
import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from tqdm import tqdm

# Import transformers for model downloading
try:
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    from huggingface_hub import hf_hub_download, snapshot_download, login
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

@dataclass
class ModelDownloadInfo:
    """Information about model download"""
    model_id: str
    model_type: str
    local_path: str
    expected_files: List[str]
    expected_size: Optional[int] = None
    checksums: Optional[Dict[str, str]] = None

class ModelCorruptionError(Exception):
    """Exception raised when model corruption is detected"""
    pass

class ModelDownloadError(Exception):
    """Exception raised when model download fails"""
    pass

class ModelManager:
    """Manages model downloading, validation, and corruption detection"""
    
    def __init__(self, models_base_dir: str = "/workspace/models/models-small-3b-6b"):
        self.models_base_dir = Path(models_base_dir)
        self.models_base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model download configurations
        self.model_configs = {
            'qwen2-vl-2b-awq': ModelDownloadInfo(
                model_id='Qwen/Qwen2-VL-2B-Instruct-AWQ',
                model_type='qwen2vl',
                local_path=str(self.models_base_dir / 'Qwen2-VL-2B-Instruct-AWQ'),
                expected_files=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'],
                expected_size=2.1 * 1024**3  # 2.1 GB
            ),
            'qwen25-omni-3b-gguf': ModelDownloadInfo(
                model_id='Qwen/Qwen2.5-Omni-3B-GGUF',
                model_type='qwen25omni',
                local_path=str(self.models_base_dir / 'Qwen2.5-Omni-3B-GGUF'),
                expected_files=['config.json', '*.gguf', 'tokenizer.json', 'tokenizer_config.json'],
                expected_size=3.2 * 1024**3  # 3.2 GB
            ),
            'qwen25-vl-3b': ModelDownloadInfo(
                model_id='Qwen/Qwen2.5-VL-3B-Instruct',
                model_type='qwen25vl',
                local_path=str(self.models_base_dir / 'Qwen2.5-VL-3B-Instruct'),
                expected_files=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'],
                expected_size=6.1 * 1024**3  # 6.1 GB
            ),
            'qwen25-vl-7b': ModelDownloadInfo(
                model_id='Qwen/Qwen2.5-VL-7B-Instruct',
                model_type='qwen25vl',
                local_path='Qwen/Qwen2.5-VL-7B-Instruct',  # Use HuggingFace cache
                expected_files=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'],
                expected_size=14.2 * 1024**3  # 14.2 GB
            )
        }
        
    def validate_model_integrity(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        Validate model integrity and detect corruption
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if model_name not in self.model_configs:
            return False, [f"Unknown model: {model_name}"]
            
        config = self.model_configs[model_name]
        issues = []
        
        # Check if model directory exists
        model_path = Path(config.local_path)
        if not model_path.exists():
            return False, [f"Model directory does not exist: {model_path}"]
            
        # Check for required files
        for expected_file in config.expected_files:
            if '*' in expected_file:
                # Handle glob patterns
                matching_files = list(model_path.glob(expected_file))
                if not matching_files:
                    issues.append(f"Missing files matching pattern: {expected_file}")
            else:
                file_path = model_path / expected_file
                if not file_path.exists():
                    issues.append(f"Missing required file: {expected_file}")
                    
        # Check file sizes (basic corruption detection)
        if config.expected_size:
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            if total_size < config.expected_size * 0.8:  # Allow 20% tolerance
                issues.append(f"Model size too small: {total_size / 1024**3:.2f}GB (expected ~{config.expected_size / 1024**3:.2f}GB)")
                
        # Check for corrupted files by attempting to load them
        try:
            if config.model_type in ['qwen2vl', 'qwen25vl']:
                # Try to load processor to check for corruption
                processor = AutoProcessor.from_pretrained(config.local_path)
                del processor
            elif config.model_type == 'qwen25omni':
                # For GGUF models, check if files are readable
                gguf_files = list(model_path.glob('*.gguf'))
                if not gguf_files:
                    issues.append("No GGUF files found")
                else:
                    # Basic file integrity check
                    for gguf_file in gguf_files:
                        if gguf_file.stat().st_size < 1024:  # Less than 1KB is suspicious
                            issues.append(f"Suspiciously small GGUF file: {gguf_file.name}")
                            
        except Exception as e:
            issues.append(f"Failed to load model components: {str(e)}")
            
        return len(issues) == 0, issues
        
    def download_model(self, model_name: str, force_redownload: bool = False) -> bool:
        """
        Download a model from HuggingFace Hub
        
        Args:
            model_name: Name of the model to download
            force_redownload: Force re-download even if model exists
            
        Returns:
            True if download successful, False otherwise
        """
        if model_name not in self.model_configs:
            self.logger.error(f"Unknown model: {model_name}")
            return False
            
        config = self.model_configs[model_name]
        model_path = Path(config.local_path)
        
        # Check if model already exists and is valid
        if not force_redownload and model_path.exists():
            is_valid, issues = self.validate_model_integrity(model_name)
            if is_valid:
                self.logger.info(f"Model {model_name} already exists and is valid")
                return True
            else:
                self.logger.warning(f"Model {model_name} exists but has issues: {issues}")
                self.logger.info(f"Re-downloading model {model_name}")
                
        # Remove existing model if it exists and we're forcing re-download
        if model_path.exists() and force_redownload:
            self.logger.info(f"Removing existing model at {model_path}")
            shutil.rmtree(model_path)
            
        try:
            self.logger.info(f"Downloading model {model_name} from {config.model_id}")
            
            # Download model using snapshot_download for complete model
            snapshot_download(
                repo_id=config.model_id,
                local_dir=config.local_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            # Validate the downloaded model
            is_valid, issues = self.validate_model_integrity(model_name)
            if not is_valid:
                raise ModelCorruptionError(f"Downloaded model is corrupted: {issues}")
                
            self.logger.info(f"Successfully downloaded and validated model {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {str(e)}")
            # Clean up partial download
            if model_path.exists():
                shutil.rmtree(model_path)
            return False
            
    def ensure_model_available(self, model_name: str, max_retries: int = 3) -> bool:
        """
        Ensure model is available and valid, with automatic re-download on corruption
        
        Args:
            model_name: Name of the model
            max_retries: Maximum number of download retries
            
        Returns:
            True if model is available and valid, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Check if model exists and is valid
                is_valid, issues = self.validate_model_integrity(model_name)
                
                if is_valid:
                    self.logger.info(f"Model {model_name} is available and valid")
                    return True
                    
                # Model is corrupted or missing, try to download
                self.logger.warning(f"Model {model_name} has issues (attempt {attempt + 1}/{max_retries}): {issues}")
                
                if self.download_model(model_name, force_redownload=True):
                    # Verify the download was successful
                    is_valid, issues = self.validate_model_integrity(model_name)
                    if is_valid:
                        self.logger.info(f"Successfully restored model {model_name}")
                        return True
                    else:
                        self.logger.error(f"Downloaded model {model_name} is still corrupted: {issues}")
                else:
                    self.logger.error(f"Failed to download model {model_name} (attempt {attempt + 1})")
                    
            except Exception as e:
                self.logger.error(f"Error ensuring model {model_name} availability (attempt {attempt + 1}): {str(e)}")
                
        self.logger.error(f"Failed to ensure model {model_name} availability after {max_retries} attempts")
        return False
        
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the local path for a model"""
        if model_name not in self.model_configs:
            return None
        return self.model_configs[model_name].local_path
        
    def list_available_models(self) -> List[str]:
        """List all available model names"""
        return list(self.model_configs.keys())
        
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model"""
        if model_name not in self.model_configs:
            return None
            
        config = self.model_configs[model_name]
        is_valid, issues = self.validate_model_integrity(model_name)
        
        return {
            'name': model_name,
            'model_id': config.model_id,
            'local_path': config.local_path,
            'exists': Path(config.local_path).exists(),
            'is_valid': is_valid,
            'issues': issues,
            'expected_size_gb': config.expected_size / 1024**3 if config.expected_size else None
        }
        
    def cleanup_corrupted_models(self) -> List[str]:
        """
        Find and remove corrupted models
        
        Returns:
            List of model names that were cleaned up
        """
        cleaned_models = []
        
        for model_name in self.model_configs.keys():
            is_valid, issues = self.validate_model_integrity(model_name)
            if not is_valid:
                config = self.model_configs[model_name]
                model_path = Path(config.local_path)
                
                if model_path.exists():
                    self.logger.info(f"Removing corrupted model: {model_name}")
                    shutil.rmtree(model_path)
                    cleaned_models.append(model_name)
                    
        return cleaned_models

def main():
    """Test the model manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Manager CLI")
    parser.add_argument("--action", type=str, required=True,
                       choices=["validate", "download", "ensure", "info", "cleanup"],
                       help="Action to perform")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--models-dir", type=str, default="/workspace/models/models-small-3b-6b",
                       help="Models directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model manager
    manager = ModelManager(args.models_dir)
    
    if args.action == "validate":
        if not args.model:
            print("Error: --model required for validate action")
            return
            
        is_valid, issues = manager.validate_model_integrity(args.model)
        print(f"Model {args.model}: {'VALID' if is_valid else 'INVALID'}")
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
                
    elif args.action == "download":
        if not args.model:
            print("Error: --model required for download action")
            return
            
        success = manager.download_model(args.model, force_redownload=True)
        print(f"Download {'SUCCESS' if success else 'FAILED'}")
        
    elif args.action == "ensure":
        if not args.model:
            print("Error: --model required for ensure action")
            return
            
        success = manager.ensure_model_available(args.model)
        print(f"Model availability: {'SUCCESS' if success else 'FAILED'}")
        
    elif args.action == "info":
        if args.model:
            info = manager.get_model_info(args.model)
            if info:
                print(json.dumps(info, indent=2))
            else:
                print(f"Unknown model: {args.model}")
        else:
            models = manager.list_available_models()
            print("Available models:")
            for model in models:
                print(f"  - {model}")
                
    elif args.action == "cleanup":
        cleaned = manager.cleanup_corrupted_models()
        if cleaned:
            print(f"Cleaned up corrupted models: {cleaned}")
        else:
            print("No corrupted models found")

if __name__ == "__main__":
    main()
