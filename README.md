# Multi-Model Parser (MMParser) - WIP

> **⚠️ Work In Progress (WIP)**  
> This project is currently under active development. Features and APIs may change without notice.

## Overview

MMParser is a comprehensive multi-model testing framework designed for evaluating various vision-language models on multimodal tasks. This framework follows top-tier research paper standards and provides modular, extensible architecture for model evaluation.

## Features

- **Multi-Model Support**: Test multiple models with consistent evaluation pipeline
- **Modular Architecture**: Separate test files for each model type
- **Comprehensive Metrics**: Detailed evaluation metrics and performance analysis
- **Flexible Configuration**: Easy-to-configure model and evaluation settings
- **Research-Grade Standards**: Following best practices from top-tier research papers

## Supported Models

### Currently Available Models

1. **Qwen2-VL-2B-AWQ** (`qwen2-vl-2b-awq`)
   - 2B parameters with AWQ quantization
   - Efficient inference with reduced memory requirements
   - Min VRAM: 4GB, Recommended: 8GB

2. **Qwen2.5-Omni-3B-GGUF** (`qwen25-omni-3b-gguf`)
   - 3B parameters with GGUF format
   - Optimized for multimodal tasks
   - Min VRAM: 6GB, Recommended: 12GB

3. **Qwen2.5-VL-3B** (`qwen25-vl-3b`)
   - 3B parameters for vision-language understanding
   - Standard safetensors format
   - Min VRAM: 6GB, Recommended: 12GB

## Project Structure

```
mmparser/
├── README.md                 # This file
├── configs/                  # Configuration files
│   ├── __init__.py
│   └── model_config.py      # Model configurations
├── tests/                    # Model-specific test files
│   ├── test_qwen2_vl_2b_awq.py
│   ├── test_qwen25_omni_3b_gguf.py
│   └── test_qwen25_vl_3b.py
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── evaluator.py         # Base evaluator class
│   └── metrics.py           # Metrics calculation
├── models/                   # Model-specific implementations
├── data/                     # Sample data and datasets
└── results/                  # Evaluation results
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA (recommended for GPU acceleration)

### Installation

```bash
# Install dependencies
pip install torch transformers accelerate
pip install Pillow opencv-python
pip install numpy pandas tqdm

# Optional: For OCR support
pip install pytesseract

# Optional: For OpenAI integration
pip install openai
```

### Basic Usage

#### Test a Single Model

```bash
# Test Qwen2-VL-2B-AWQ model with LongDocURL dataset
python tests/test_qwen2_vl_2b_awq.py \
    --data-path ../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl \
    --task understanding \
    --num-samples 100

# Test Qwen2.5-VL-3B model with LongDocURL dataset
python tests/test_qwen25_vl_3b.py \
    --data-path ../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl \
    --task reasoning \
    --num-samples 50
```

#### Batch Evaluation

```bash
# Run batch evaluation (coming soon)
python batch_eval.py --config configs/batch_config.json
```

## Configuration

### Model Configuration

Models are configured in `configs/model_config.py`:

```python
from configs.model_config import get_model_config, get_available_models

# Get available models
models = get_available_models()
print(models)  # ['qwen2-vl-2b-awq', 'qwen25-omni-3b-gguf', 'qwen25-vl-3b']

# Get model configuration
config = get_model_config('qwen2-vl-2b-awq')
print(config.description)
```

### Evaluation Configuration

Each test file accepts various configuration options:

- `--data-path`: Path to dataset (JSONL file or directory)
- `--task`: Task type (understanding, reasoning, locating)
- `--num-samples`: Number of samples to evaluate (-1 for all)
- `--output-dir`: Output directory for results
- `--device`: Device to use (auto, cpu, cuda)
- `--max-new-tokens`: Maximum tokens to generate
- `--temperature`: Generation temperature
- `--top-p`: Top-p sampling parameter

## Dataset Format

The framework uses the LongDocURL dataset format. The data loader automatically converts LongDocURL format to our standard format:

### LongDocURL Original Format
```json
{
    "question_id": "free_gpt4o_4045421_5_34_5",
    "doc_no": "4045421",
    "total_pages": 59,
    "start_end_idx": [5, 34],
    "question_type": "calculate",
    "question": "What is the percentage decrease of adolescent pregnancy rates among females aged 18-19 from 2013 to 2017?",
    "answer": 23.73,
    "detailed_evidences": "To calculate the percentage decrease...",
    "evidence_pages": [6],
    "evidence_sources": ["Figure"],
    "answer_format": "Float",
    "task_tag": "Reasoning",
    "images": ["/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4045/4045421_4.png"],
    "pdf_path": "/data/oss_bucket_0/achao.dc/public_datasets/ccpdf_zip/4000-4999/4045421.pdf",
    "subTask": ["SP_Figure_Reasoning"]
}
```

### Converted Standard Format
```json
{
    "question_id": "free_gpt4o_4045421_5_34_5",
    "question": "What is the percentage decrease of adolescent pregnancy rates among females aged 18-19 from 2013 to 2017?",
    "images": ["../data/LongDocURL/pdf_pngs/4045/4045421_4.png"],
    "answer": 23.73,
    "task": "reasoning"
}
```

## Output Format

### Summary Results (`*_summary.json`)

```json
{
    "experiment_id": "20241201_143022_qwen2-vl-2b-awq_understanding",
    "timestamp": "2024-12-01T14:30:22",
    "model_name": "qwen2-vl-2b-awq",
    "metrics": {
        "accuracy": 0.85,
        "total_correct": 85,
        "total_samples": 100
    },
    "performance": {
        "total_time": 120.5,
        "avg_time_per_sample": 1.205
    }
}
```

### Detailed Results (`*_detailed.json`)

```json
{
    "experiment_id": "20241201_143022_qwen2-vl-2b-awq_understanding",
    "detailed_results": [
        {
            "question_id": 0,
            "question": "What company's financial statement is this?",
            "images": ["/path/to/image1.png"],
            "ground_truth": "Apple Inc.",
            "model_response": "This appears to be Apple Inc.'s financial statement...",
            "is_correct": true,
            "evaluation_time": 1.2
        }
    ]
}
```

## Development Status

### ✅ Completed Features

- [x] Basic project structure
- [x] Model configuration system
- [x] Qwen2-VL-2B-AWQ test implementation
- [x] Base evaluator framework
- [x] Results saving and logging

### 🚧 In Progress

- [ ] Qwen2.5-Omni-3B-GGUF test implementation
- [ ] Qwen2.5-VL-3B test implementation
- [ ] Batch evaluation system
- [ ] Comprehensive metrics calculator
- [ ] GPU memory optimization
- [ ] Multi-GPU support

### 📋 Planned Features

- [ ] Additional model support (LLaVA, CLIP, etc.)
- [ ] Advanced metrics (BLEU, ROUGE, etc.)
- [ ] Visualization tools
- [ ] Web interface for results
- [ ] Distributed evaluation
- [ ] Model comparison tools

## Contributing

This project is currently in WIP status. Contributions are welcome but please note:

1. The codebase is under active development
2. APIs and interfaces may change frequently
3. Please check with maintainers before major contributions
4. Follow the existing code style and documentation standards

## Hardware Requirements

### Minimum Requirements

- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 20GB+ free space
- **GPU**: Optional, but recommended for faster inference

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ free space
- **GPU**: NVIDIA GPU with 12GB+ VRAM

### Model-Specific Requirements

| Model | Min VRAM | Recommended VRAM | Notes |
|-------|----------|------------------|-------|
| Qwen2-VL-2B-AWQ | 4GB | 8GB | Quantized model |
| Qwen2.5-Omni-3B-GGUF | 6GB | 12GB | GGUF format |
| Qwen2.5-VL-3B | 6GB | 12GB | Standard format |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use smaller models
   - Enable gradient checkpointing

2. **Model Loading Error**
   - Check model path in configuration
   - Verify model files exist
   - Check disk space

3. **CUDA Issues**
   - Verify CUDA installation
   - Check GPU memory availability
   - Use CPU fallback if needed

### Getting Help

- Check the logs in the output directory
- Verify model configurations
- Ensure all dependencies are installed
- Check hardware requirements

## License

This project is part of the ICLR research framework and follows the same license terms.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{mmparser2024,
  title={MMParser: Multi-Model Parser for Vision-Language Evaluation},
  author={ICLR Research Team},
  year={2024},
  note={Work in Progress}
}
```

---

**⚠️ Note**: This is a work-in-progress project. Features and APIs are subject to change. Use with caution in production environments.
