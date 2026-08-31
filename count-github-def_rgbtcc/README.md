# DEF-rgbtcc Computer Vision - Multimodal RGB-T Crowd Counting

A high-performance, production-ready computer vision framework for **Dual-Modality (RGB-Thermal) Crowd Counting** based on the **DEF-rgbtcc** architecture (*Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion*, ArXiv 2509.17079).

---

## 🌟 Features & Highlights

- **Dual-Stream Synchronized I/O**: Multithreaded OpenCV background readers (`DualStreamVideoReader`) for synchronized RGB and Thermal camera feeds.
- **DEF-rgbtcc Architecture**: Dual-modality feature extraction with shared VGG-19 backbone, Spatially Modulated Attention (SMA) Transformer, Adaptive Cross-Modal Fusion (ACMF), and 2D density map regression.
- **Multi-Format RTX Acceleration**: TensorRT FP16/FP32 (`.trt`), SafeTensors (`.safetensors`), PyTorch (`.pth`), ONNX (`.onnx`), and HuggingFace integration (`ilessio-aiflowlab/DEF-rgbtcc`).
- **Density Heatmap Visualization**: Automated rendering of colored density heatmaps (`JET`, `VIRIDIS`, `HOT`, `TURBO`, `INFERNO`, `PLASMA`) overlaid with live count banners and frame statistics.
- **Adaptive Memory Management**: Automatic CUDA OOM handling with batch halving and memory cache cleaning.
- **Full Telemetry & Export**: Structured CSV time-series (`frame_counts.csv`), JSON analytics summaries (`summary.json`), and MLflow experiment tracking.

---

## ⚡ Quick Start

### 1. Execute Pipeline with Local `.venv`

```bash
# Run with Day RGBT Configuration
python run.py --config data_rgbt_day.yaml

# Run with Night RGBT Configuration
python run.py --config data_rgbt_night.yaml
```

### 2. Run Test Suite

```bash
pytest app/tests/
```

---

## 🏗 System Architecture

```
app/
├── head_counting/
│   ├── config.py         # PipelineConfig & RGBT Dataclasses
│   ├── video.py          # DualStreamVideoReader & DualStreamVideoWriterWrapper
│   ├── model.py          # DEFModelHandler & TensorRT Priority Resolution
│   ├── pipeline.py       # CountingPipeline Orchestrator
│   └── tracker.py        # MLflow Tracker & Telemetry Manager
├── tests/                # 13 Automated Pytest Units & E2E Integration Tests
├── data_rgbt_day.yaml    # Production Configuration (Day)
├── data_rgbt_night.yaml  # Production Configuration (Night)
└── run.py                # Command Line Interface
```

---

## 📖 Reference Architecture (DEF-rgbtcc)

- **Backbone**: Shared VGG-19
- **Encoder**: Spatially Modulated Attention (SMA) Transformer
- **Fusion**: Adaptive Cross-Modal Fusion (ACMF)
- **Output**: Density map regression (`density_map`) & total population count (`count`)
- **ANIMA Ecosystem**: Integrated with ANIMA Defense Module ecosystem (Wave 8 - Products: ORACLE, ATLAS, NEMESIS).