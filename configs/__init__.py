"""
Configuration module for multi-model testing framework
"""

from .model_config import (
    AVAILABLE_MODELS,
    get_model_config,
    get_available_models,
    validate_model_path,
    get_model_info,
    ModelConfig
)

__all__ = [
    'AVAILABLE_MODELS',
    'get_model_config',
    'get_available_models', 
    'validate_model_path',
    'get_model_info',
    'ModelConfig'
]
