# Model Management and Corruption Detection

This document describes the automatic model corruption detection and re-download functionality added to the MMParser framework.

## Overview

The system now includes comprehensive model management capabilities that automatically detect corrupted models and re-download them when necessary. This ensures reliable model loading and prevents evaluation failures due to corrupted model files.

## Key Features

### 1. Automatic Corruption Detection
- **File Integrity Checks**: Validates that all required model files exist
- **Size Validation**: Checks if model files have expected sizes
- **Loading Tests**: Attempts to load model components to detect corruption
- **Pattern Recognition**: Identifies common corruption indicators in error messages

### 2. Automatic Re-download
- **Smart Retry Logic**: Automatically re-downloads corrupted models
- **Cache Management**: Clears corrupted cache files before re-download
- **Progress Tracking**: Shows download progress with retry attempts
- **Validation**: Verifies downloaded models before use

### 3. Integration with Existing Code
- **Seamless Integration**: Works with existing evaluation scripts
- **Backward Compatibility**: Falls back to basic validation if ModelManager unavailable
- **Error Handling**: Graceful handling of download failures

## Components

### ModelManager Class (`utils/model_manager.py`)

The core component that handles all model management operations:

```python
from utils.model_manager import ModelManager

manager = ModelManager()

# Validate model integrity
is_valid, issues = manager.validate_model_integrity("qwen25-vl-3b")

# Ensure model is available (with auto re-download)
success = manager.ensure_model_available("qwen25-vl-3b")

# Get model information
info = manager.get_model_info("qwen25-vl-3b")
```

### Model Configuration Integration (`configs/model_config.py`)

Enhanced model configuration with corruption detection:

```python
from configs.model_config import ensure_model_available, validate_model_path

# Validate model (with corruption detection)
is_valid = validate_model_path("qwen25-vl-3b")

# Ensure model availability (with auto re-download)
success = ensure_model_available("qwen25-vl-3b")
```

### Base Evaluator Integration (`utils/evaluator.py`)

All evaluators now have access to model management:

```python
class MyEvaluator(BaseEvaluator):
    def load_model(self):
        # Ensure model is available before loading
        if not self.ensure_model_availability(self.config.model_name):
            raise RuntimeError("Failed to ensure model availability")
        
        # Load model with retry logic
        # ... existing loading code ...
```

## Supported Models

The system currently supports the following models with automatic corruption detection:

- `qwen2-vl-2b-awq`: Qwen2-VL-2B with AWQ quantization
- `qwen25-omni-3b-gguf`: Qwen2.5-Omni-3B with GGUF format
- `qwen25-vl-3b`: Qwen2.5-VL-3B standard model
- `qwen25-vl-7b`: Qwen2.5-VL-7B standard model

## Usage Examples

### 1. Basic Model Validation

```python
from utils.model_manager import ModelManager

manager = ModelManager()

# Check if a model is valid
is_valid, issues = manager.validate_model_integrity("qwen25-vl-3b")
if not is_valid:
    print(f"Model has issues: {issues}")
```

### 2. Automatic Model Recovery

```python
# This will automatically re-download if corrupted
success = manager.ensure_model_available("qwen25-vl-3b")
if success:
    print("Model is ready to use")
else:
    print("Failed to ensure model availability")
```

### 3. Integration in Evaluation Scripts

The evaluation scripts now automatically handle model corruption:

```python
# In any evaluation script
def load_model(self):
    # This will automatically re-download corrupted models
    if not self.ensure_model_availability(self.config.model_name):
        raise RuntimeError("Failed to ensure model availability")
    
    # Load model with retry logic
    for attempt in range(max_retries):
        try:
            self.model = AutoModel.from_pretrained(...)
            break
        except Exception as e:
            if "corrupted" in str(e).lower() and attempt < max_retries - 1:
                self.logger.warning("Detected corruption, re-downloading...")
                self.ensure_model_availability(self.config.model_name)
            else:
                raise
```

## Command Line Interface

The ModelManager includes a CLI for testing and management:

```bash
# Validate a model
python utils/model_manager.py --action validate --model qwen25-vl-3b

# Download a model
python utils/model_manager.py --action download --model qwen25-vl-3b

# Ensure model availability
python utils/model_manager.py --action ensure --model qwen25-vl-3b

# Get model information
python utils/model_manager.py --action info --model qwen25-vl-3b

# Clean up corrupted models
python utils/model_manager.py --action cleanup
```

## Testing

Run the test suite to verify functionality:

```bash
python test_model_manager.py
```

This will test:
- Model validation
- Model availability
- Config integration
- Corruption detection

## Error Handling

The system handles various error scenarios:

### 1. Network Issues
- Automatic retry with exponential backoff
- Graceful handling of connection timeouts
- Resume interrupted downloads

### 2. Disk Space Issues
- Checks available disk space before download
- Cleans up partial downloads on failure
- Provides clear error messages

### 3. Permission Issues
- Handles read/write permission errors
- Provides guidance for fixing permissions
- Falls back to user cache if needed

## Configuration

### Model Download Settings

Models are configured in `utils/model_manager.py`:

```python
self.model_configs = {
    'qwen25-vl-3b': ModelDownloadInfo(
        model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        model_type='qwen25vl',
        local_path=str(self.models_base_dir / 'Qwen2.5-VL-3B-Instruct'),
        expected_files=['config.json', 'model.safetensors', 'tokenizer.json'],
        expected_size=6.1 * 1024**3  # 6.1 GB
    ),
    # ... other models
}
```

### Retry Settings

Configure retry behavior:

```python
# In ensure_model_available()
max_retries = 3  # Maximum download attempts
```

## Logging

The system provides comprehensive logging:

```
2024-01-15 10:30:15 - INFO - Ensuring model availability for qwen25-vl-3b
2024-01-15 10:30:15 - WARNING - Model qwen25-vl-3b has issues: ['Missing required file: model.safetensors']
2024-01-15 10:30:15 - INFO - Re-downloading model qwen25-vl-3b
2024-01-15 10:30:15 - INFO - Downloading model qwen25-vl-3b from Qwen/Qwen2.5-VL-3B-Instruct
2024-01-15 10:35:42 - INFO - Successfully downloaded and validated model qwen25-vl-3b
```

## Best Practices

### 1. Model Validation
- Always validate models before use
- Check for corruption after downloads
- Monitor disk space during downloads

### 2. Error Handling
- Implement proper retry logic
- Log all download attempts
- Provide user feedback on progress

### 3. Performance
- Use local model paths when possible
- Cache validation results
- Parallel downloads for multiple models

## Troubleshooting

### Common Issues

1. **Download Failures**
   - Check internet connection
   - Verify HuggingFace Hub access
   - Check disk space

2. **Validation Failures**
   - Ensure all required files exist
   - Check file permissions
   - Verify model format compatibility

3. **Loading Errors**
   - Check CUDA/CPU compatibility
   - Verify model dependencies
   - Check memory requirements

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements include:

1. **Checksum Validation**: MD5/SHA256 verification of model files
2. **Incremental Downloads**: Resume partial downloads
3. **Model Compression**: Support for compressed model formats
4. **Cloud Storage**: Integration with cloud storage providers
5. **Model Versioning**: Support for model version management

## Dependencies

The model management system requires:

- `transformers>=4.35.0`
- `huggingface_hub>=0.19.0`
- `torch>=2.0.0`
- `tqdm>=4.65.0`
- `requests>=2.31.0`

These are included in the updated `requirements.txt`.
