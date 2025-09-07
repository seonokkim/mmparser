# Multi-Model Parser (MMParser)

> **🚀 Active Development**  
> This project is actively maintained and continuously improved with new features and model support.

## Overview

MMParser is a comprehensive multi-model testing framework designed for evaluating various vision-language models on multimodal tasks using the LongDocURL dataset. The framework provides standardized evaluation pipelines for comparing different models across understanding, reasoning, and locating tasks.

## Features

- **Multi-Model Support**: Test multiple models with consistent evaluation pipeline
- **Modular Architecture**: Separate evaluation scripts for each model type
- **Comprehensive Metrics**: Detailed evaluation metrics and performance analysis
- **Flexible Configuration**: Easy-to-configure model and evaluation settings
- **Batch Evaluation**: Run multiple experiments with different configurations
- **LongDocURL Dataset**: Built-in support for LongDocURL dataset format
- **Standardized Output**: Consistent result formats across all models

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

4. **Qwen2.5-VL-7B** (`qwen25-vl-7b`)
   - 7B parameters for vision-language understanding
   - Standard safetensors format
   - Min VRAM: 8GB, Recommended: 16GB

5. **Qwen2-VL-7B** (`qwen2-vl-7b`)
   - 7B parameters for vision-language understanding
   - Standard safetensors format
   - Min VRAM: 8GB, Recommended: 16GB

6. **LLaVA-7B** (`llava-7b`)
   - 7B parameters LLaVA model
   - Vision-language understanding
   - Min VRAM: 8GB, Recommended: 16GB

7. **LLaVA-Next-7B** (`llava-next-7b`)
   - 7B parameters LLaVA-Next model
   - Enhanced vision-language capabilities
   - Min VRAM: 8GB, Recommended: 16GB

8. **LLaVA-OneVision-7B** (`llava-onevision-7b`)
   - 7B parameters LLaVA-OneVision model
   - Specialized vision understanding
   - Min VRAM: 8GB, Recommended: 16GB

9. **LLaVA-OneVision-Chat-7B** (`llava-onevision-chat-7b`)
   - 7B parameters LLaVA-OneVision-Chat model
   - Conversational vision understanding
   - Min VRAM: 8GB, Recommended: 16GB

10. **LLaVA-Next-Interleave-7B** (`llava-next-interleave-7b`)
    - 7B parameters LLaVA-Next-Interleave model
    - Interleaved vision-language processing
    - Min VRAM: 8GB, Recommended: 16GB

11. **LLaMA-3-8B** (`llama-3-8b`)
    - 8B parameters LLaMA-3 model
    - General language understanding
    - Min VRAM: 10GB, Recommended: 20GB

12. **LLaMA-32-11B** (`llama-32-11b`)
    - 11B parameters LLaMA-32 model
    - Enhanced language understanding
    - Min VRAM: 12GB, Recommended: 24GB

13. **InternVL3-9B** (`internvl3-9b`)
    - 9B parameters InternVL3 model
    - Vision-language understanding
    - Min VRAM: 10GB, Recommended: 20GB

14. **Qwen3-8B** (`qwen3-8b`)
    - 8B parameters Qwen3 model
    - General language understanding
    - Min VRAM: 10GB, Recommended: 20GB

## Project Structure

```
mmparser/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── batch_eval.py            # Batch evaluation script
├── configs/                  # Configuration files
│   ├── __init__.py
│   ├── model_config.py      # Model configurations
│   ├── batch_config_example.json  # Example batch config
│   ├── qwen25_vl_3b_single_test.json  # Single test config
│   ├── qwen25_vl_3b_test.json  # Test config
│   └── *.json               # Model-specific configs
├── eval/                     # Model-specific evaluation scripts
│   ├── eval_qwen2_vl_2b_awq.py
│   ├── eval_qwen25_omni_3b_gguf.py
│   ├── eval_qwen25_vl_3b.py
│   ├── eval_qwen25_vl_7b.py
│   ├── eval_qwen2_vl_7b.py
│   ├── eval_llava_7b.py
│   ├── eval_llava_next_7b.py
│   ├── eval_llava_onevision_7b.py
│   ├── eval_llava_onevision_chat_7b.py
│   ├── eval_llava_next_interleave_7b.py
│   ├── eval_llama_3_8b.py
│   ├── eval_llama_32_11b_11b.py
│   ├── eval_internvl3_9b_9b.py
│   └── eval_qwen3_8b_8b.py
├── tests/                    # Unit tests
│   └── test_data_loader.py
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── evaluator.py         # Base evaluator class
│   ├── data_loader.py       # LongDocURL data loader
│   └── metrics.py           # Metrics calculation
└── venv/                     # Virtual environment (local)
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA (recommended for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mmparser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: For OCR support
pip install pytesseract

# Optional: For OpenAI integration
pip install openai
```

### Basic Usage

#### Test a Single Model

```bash
# Test Qwen2-VL-2B-AWQ model with LongDocURL dataset
python eval/eval_qwen2_vl_2b_awq.py \
    --data-path ../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl \
    --task understanding \
    --num-samples 100

# Test Qwen2.5-VL-3B model with LongDocURL dataset
python eval/eval_qwen25_vl_3b.py \
    --data-path ../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl \
    --task reasoning \
    --num-samples 50

# Test LLaVA-7B model
python eval/eval_llava_7b.py \
    --data-path ../data/LongDocURL/LongDocURL_public_with_subtask_category_10pct.jsonl \
    --task understanding \
    --num-samples 25
```

#### Batch Evaluation

```bash
# Create example batch configuration
python batch_eval.py --create-example

# Run batch evaluation with custom config
python batch_eval.py --config configs/batch_config_example.json

# Run batch evaluation for specific model only
python batch_eval.py --config configs/batch_config_example.json --model qwen2-vl-2b-awq

# Run batch evaluation with custom output
python batch_eval.py --config configs/batch_config_example.json --output results/my_batch_results.json

# Run single test configuration
python batch_eval.py --config configs/qwen25_vl_3b_single_test.json

# Run test configuration with specific output
python batch_eval.py --config configs/qwen25_vl_3b_test.json --output results/test_results.json
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
- [x] Qwen2-VL-2B-AWQ evaluation implementation
- [x] Qwen2.5-Omni-3B-GGUF evaluation implementation
- [x] Qwen2.5-VL-3B evaluation implementation
- [x] Qwen2.5-VL-7B evaluation implementation
- [x] Qwen2-VL-7B evaluation implementation
- [x] LLaVA-7B evaluation implementation
- [x] LLaVA-Next-7B evaluation implementation
- [x] LLaVA-OneVision-7B evaluation implementation
- [x] LLaVA-OneVision-Chat-7B evaluation implementation
- [x] LLaVA-Next-Interleave-7B evaluation implementation
- [x] LLaMA-3-8B evaluation implementation
- [x] LLaMA-32-11B evaluation implementation
- [x] InternVL3-9B evaluation implementation
- [x] Qwen3-8B evaluation implementation
- [x] Base evaluator framework
- [x] LongDocURL data loader
- [x] Results saving and logging
- [x] Batch evaluation system
- [x] Comprehensive metrics calculator
- [x] GPU memory optimization
- [x] Enhanced batch evaluation with filtering
- [x] Configuration file management
- [x] Comprehensive .gitignore setup

### 🚧 In Progress

- [ ] Multi-GPU support
- [ ] Advanced metrics (BLEU, ROUGE, etc.)
- [ ] Performance optimization
- [ ] Error handling improvements

### 📋 Planned Features

- [ ] Additional model support (CLIP, BLIP, etc.)
- [ ] Visualization tools
- [ ] Web interface for results
- [ ] Distributed evaluation
- [ ] Model comparison tools
- [ ] Automated benchmarking
- [ ] Result analysis dashboard

## Contributing

This project is actively maintained and contributions are welcome! Please note:

1. The codebase is under active development
2. APIs and interfaces may change with new features
3. Please check with maintainers before major contributions
4. Follow the existing code style and documentation standards
5. Test your changes thoroughly before submitting

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
| Qwen2.5-VL-7B | 8GB | 16GB | Standard format |
| Qwen2-VL-7B | 8GB | 16GB | Standard format |
| LLaVA-7B | 8GB | 16GB | Vision-language model |
| LLaVA-Next-7B | 8GB | 16GB | Enhanced LLaVA |
| LLaVA-OneVision-7B | 8GB | 16GB | Specialized vision |
| LLaVA-OneVision-Chat-7B | 8GB | 16GB | Conversational |
| LLaVA-Next-Interleave-7B | 8GB | 16GB | Interleaved processing |
| LLaMA-3-8B | 10GB | 20GB | Language model |
| LLaMA-32-11B | 12GB | 24GB | Large language model |
| InternVL3-9B | 10GB | 20GB | Vision-language model |
| Qwen3-8B | 10GB | 20GB | Language model |

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

This project follows standard open source license terms.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{mmparser2025,
  title={MMParser: Multi-Model Parser for Vision-Language Evaluation},
  author={Riley Kim},
  year={2025},
  note={Work in Progress}
}
```

---

**📝 Note**: This project is actively maintained and continuously improved. For production use, please test thoroughly and check for the latest updates.
