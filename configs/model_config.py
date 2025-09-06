#!/usr/bin/env python3
"""
Model Configuration for Multi-Model Testing Framework

This module contains configurations for different models available in the system,
following top-tier research paper standards for model evaluation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Base model directory (relative to project root)
MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "models-small-3b-6b")

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    type: str
    model_class: str
    processor_class: str
    model_path: str
    parameters: str
    min_vram: int
    recommended_vram: int
    description: str
    supported_tasks: list
    generation_config: Dict[str, Any]

# Available models configuration
AVAILABLE_MODELS = {
    'qwen2-vl-2b-awq': ModelConfig(
        name='qwen2-vl-2b-awq',
        type='qwen',
        model_class='Qwen2VLForConditionalGeneration',
        processor_class='AutoProcessor',
        model_path=os.path.join(MODEL_BASE_DIR, 'Qwen2-VL-2B-Instruct-AWQ'),
        parameters='2B',
        min_vram=4,
        recommended_vram=8,
        description='Qwen2-VL-2B model with AWQ quantization for efficient inference',
        supported_tasks=['understanding', 'reasoning', 'locating'],
        generation_config={
            'max_new_tokens': 512,
            'temperature': 0.3,
            'top_p': 0.9,
            'do_sample': True
        }
    ),
    
    'qwen25-omni-3b-gguf': ModelConfig(
        name='qwen25-omni-3b-gguf',
        type='qwen',
        model_class='Qwen2_5_OmniForConditionalGeneration',
        processor_class='AutoProcessor',
        model_path=os.path.join(MODEL_BASE_DIR, 'Qwen2.5-Omni-3B-GGUF'),
        parameters='3B',
        min_vram=6,
        recommended_vram=12,
        description='Qwen2.5-Omni-3B model with GGUF format for multimodal tasks',
        supported_tasks=['understanding', 'reasoning', 'locating'],
        generation_config={
            'max_new_tokens': 512,
            'temperature': 0.3,
            'top_p': 0.9,
            'do_sample': True
        }
    ),
    
    'qwen25-vl-3b': ModelConfig(
        name='qwen25-vl-3b',
        type='qwen',
        model_class='Qwen2_5_VLForConditionalGeneration',
        processor_class='AutoProcessor',
        model_path=os.path.join(MODEL_BASE_DIR, 'Qwen2.5-VL-3B-Instruct'),
        parameters='3B',
        min_vram=6,
        recommended_vram=12,
        description='Qwen2.5-VL-3B model for vision-language understanding tasks',
        supported_tasks=['understanding', 'reasoning', 'locating'],
        generation_config={
            'max_new_tokens': 512,
            'temperature': 0.3,
            'top_p': 0.9,
            'do_sample': True
        }
    )
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_name]

def get_available_models() -> list:
    """Get list of available model names"""
    return list(AVAILABLE_MODELS.keys())

def validate_model_path(model_name: str) -> bool:
    """Validate that model files exist"""
    config = get_model_config(model_name)
    model_path = Path(config.model_path)
    return model_path.exists() and any(model_path.glob("*.safetensors")) or any(model_path.glob("*.gguf"))

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a model"""
    config = get_model_config(model_name)
    return {
        'name': config.name,
        'type': config.type,
        'parameters': config.parameters,
        'min_vram': config.min_vram,
        'recommended_vram': config.recommended_vram,
        'description': config.description,
        'supported_tasks': config.supported_tasks,
        'model_path': config.model_path,
        'path_exists': validate_model_path(model_name)
    }
