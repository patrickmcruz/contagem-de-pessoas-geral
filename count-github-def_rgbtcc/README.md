# YOLO26 Computer Vision - Notebooks & Analysis

A complete, production-ready framework for training, testing, and validating YOLO26 object detection and pose estimation models.

## Quick Start

All notebooks are organized by function in the `notebooks/` directory:

```bash
cd notebooks/

# Setup (first time only)
cd setup/
# Run: setup-density-hardware-analysis.ipynb
# Run: test-setup-01.ipynb

# Train a model
cd ../training/
# Choose and run one of the training notebooks

# Test & validate
cd ../testing/
# Run test suites

# Final check
cd ../validation/
# Run: test-quick-validation.ipynb
```

## Notebooks Organization

### Four Categories, One Purpose

| Category | Purpose | Notebooks | Time |
|----------|---------|-----------|------|
| **setup/** | Environment & hardware analysis | 2 | 15 min |
| **training/** | Model training workflows | 4 | 60-120 min |
| **testing/** | Performance testing & optimization | 3 | 30-40 min |
| **validation/** | End-to-end validation | 1 | 1 min |

**Total**: 10 notebooks, all production-ready

### Where to Start

1. **First time?** → Read `notebooks/README.md`
2. **Quick overview?** → Read `notebooks/setup/01-SETUP-README.md`
3. **Training?** → Read `notebooks/training/02-TRAINING-README.md`
4. **Testing?** → Read `notebooks/testing/03-TESTING-README.md`
5. **Validation?** → Read `notebooks/validation/04-VALIDATION-README.md`

## Repository Structure

```
notebooks/                    # All Jupyter notebooks organized by function
├── test-setup-11-heads-counting-enhanced.ipynb # Jupyter Notebook para detecção e contagem de cabeças
├── README.md               # Master guide (START HERE)
├── setup/                  # Environment setup & hardware analysis
├── training/               # Model training workflows  
├── testing/                # Comprehensive test suites
├── validation/             # Quick sanity checks
└── models/                 # Pre-trained YOLO26 models
    ├── yolo26n.pt
    ├── yolo26m.pt
    └── yolo26x.pt

assets/                      # Example images and resources
automation/                  # Deployment & CI/CD scripts
requirements.txt            # Python dependencies
```

## Key Features

✓ **Well-Organized** - 4 categories by function  
✓ **Production-Ready** - Tested and validated  
✓ **Self-Contained** - Each notebook includes docs  
✓ **Scalable** - Easy to add new models  
✓ **Documented** - 4 category READMEs + inline docstrings  
✓ **Configurable** - YAML configs for each category  

## Documentation Map

Each category has complete documentation:

- **README** - Procedures, prerequisites, troubleshooting
- **CONFIG.yaml** - Parameters, dependencies, hardware requirements
- **Notebooks** - Cell-level docstrings and markdown

### Category READMEs

- `notebooks/setup/01-SETUP-README.md` - Setup procedures
- `notebooks/training/02-TRAINING-README.md` - Training workflows
- `notebooks/testing/03-TESTING-README.md` - Test suites
- `notebooks/validation/04-VALIDATION-README.md` - Validation checks

## Getting Started

### 1. Environment Setup (15 min)

```bash
cd notebooks/setup/
jupyter notebook
# Open: setup-density-hardware-analysis.ipynb
# Follow: 01-SETUP-README.md
```

### 2. Choose Your Path

**Train a New Model** (60-120 min)
```bash
cd notebooks/training/
# See: 02-TRAINING-README.md
# Choose a training notebook and prepare your dataset
```

**Quick Validation** (1 min)
```bash
cd notebooks/validation/
jupyter notebook test-quick-validation.ipynb
```

**Run Tests** (30 min)
```bash
cd notebooks/testing/
# See: 03-TESTING-README.md for test suite details
```

## Python Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended 8GB+ VRAM)
- See: `requirements.txt`

## Pre-trained Models

Available models (stored in `notebooks/models/`):

- **yolo26n.pt** - Nano model (5.3 MB, ~50ms inference)
- **yolo26m.pt** - Medium model (42.2 MB, ~100ms inference)
- **yolo26x.pt** - X-Large model (113.2 MB, ~200ms inference)

## Documentation Structure

```
📖 Documentation Levels:

1. Project Level (README.md) - You are here
2. Category Level (notebooks/README.md) - Overview of all notebooks
3. Category Guides (notebooks/*/README.md) - Detailed procedures
4. Configuration (notebooks/*/CONFIG.yaml) - Technical parameters
5. Code Level - In-notebook docstrings and comments
```

## Common Workflows

### Complete Pipeline (Setup → Train → Test → Deploy)
```
setup/ → training/ → testing/ → validation/ → production
```

### Quick Validation Only
```
validation/ → decision (deploy or iterate)
```

### Continuous Improvement (Train → Test → Optimize → Re-train)
```
setup/ → training/ → testing/ → re-train → validation/
```

### CI/CD Pipeline
```
validation/ → pass/fail → deploy/block
```

## Troubleshooting

**Setup Issues?** → `notebooks/setup/01-SETUP-README.md`  
**Training Issues?** → `notebooks/training/02-TRAINING-README.md`  
**Testing Issues?** → `notebooks/testing/03-TESTING-README.md`  
**Validation Issues?** → `notebooks/validation/04-VALIDATION-README.md`  

Each guide includes a dedicated troubleshooting section.

## File Organization Principles

✓ **Function-based** - Organized by purpose (not complexity)  
✓ **Self-contained** - Each category has everything needed  
✓ **Well-documented** - Multi-level documentation  
✓ **No duplicates** - Clean, organized structure  
✓ **Scalable** - Easy to add more notebooks  

## Project Status

- **Notebooks**: 10/10 organized
- **Documentation**: Complete
- **Structure**: Production-ready
- **Last Updated**: 2026-05-26

## Next Steps

1. Read `notebooks/README.md` for a complete overview
2. Choose a category (setup, training, testing, or validation)
3. Follow the category README
4. Run notebooks in order
5. Check `notebooks/validation/` for final validation

---

**Start here**: `notebooks/README.md`

## References informations

### DEF-rgbtcc: RGB-T Crowd Counting

Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion.

Paper: ArXiv 2509.17079
Architecture

    Backbone: Shared VGG-19
    Encoder: Spatially Modulated Attention (SMA) Transformer
    Fusion: Adaptive Cross-Modal Fusion (ACMF)
    Output: Density map regression

Available Formats

    model.pth — PyTorch state dict
    model.safetensors — SafeTensors format
    model.onnx — ONNX (opset 17)
    model_fp16.trt — TensorRT FP16
    model_fp32.trt — TensorRT FP32

Usage

from def_rgbtcc.serve import RGBTCCInference

model = RGBTCCInference("model.pth")
result = model.predict(rgb_image, thermal_image)
print(f"Count: {result['count']:.1f}")

ANIMA Module

Part of the ANIMA Defense Module ecosystem (Wave 8). Products: ORACLE, ATLAS, NEMESIS