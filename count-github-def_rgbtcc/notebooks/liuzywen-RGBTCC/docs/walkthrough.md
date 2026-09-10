# Walkthrough: Implementação e Validação do Modelo liuzywen-RGBTCC (BMVC 2022)

Este walkthrough documenta a implementação, execução ponta a ponta e validação experimental do pipeline dedicado ao segundo modelo: **liuzywen-RGBTCC** (*RGB-T Multi-Modal Crowd Counting Based on Transformer*, Zhengyi Liu et al., BMVC 2022 / arXiv:2301.03033v1) na branch [`feat/liuzywen-rgbtcc-pipeline`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc).

---

## 1. O que foi Desenvolvido

### A. Separação Modular e Isolamento Estrutural
- **Diretório Dedicado:** [`notebooks/liuzywen-RGBTCC/`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/)
- Garante independência total em relação ao modelo `DEF-rgbtcc`, evitando contaminação cruzada de tensores, normalizações ou parâmetros.

### B. Arquitetura Neural Modular em PyTorch (`models/`)
1. **Backbone PVTv2 (`models/pvt_v2.py`):**
   - Implementação do *Pyramid Vision Transformer v2* com 4 estágios hierárquicos:
     - $F^1$: Stride 4, Dim 64
     - $F^2$: Stride 8, Dim 128
     - $F^3$: Stride 16, Dim 320
     - $F^4$: Stride 32, Dim 512
   - Mecanismo *Linear Spatial Reduction Attention* (SRA) e *MixFFN* com convoluções profundas (*depth-wise*).
2. **MSTTrans - Count-Guided Multi-Scale Token Transformer (`models/liuzywen_rgbtcc_net.py`):**
   - Injeção de token global treinável de contagem $F_{count} \in \mathbb{R}^{1 \times C}$.
   - Fusão simultânea em 3 escalas de tokens: $N^2$ (inicial), $N$ (intermediária) e $1$ (global).
   - Processamento paralelo por 3 ramos de MHSA, restauração linear de dimensões e fusão residual via MLP.
3. **MSDTrans - Modal-Guided Count Enhancement (`models/liuzywen_rgbtcc_net.py`):**
   - Query térmica $[G_t, G_{count}]$ reforçada sob o contexto multiescala das features visuais $\{G_r, F_r^3, F_r^2, F_r^1\}$.
4. **Density Regression Head & Factory (`models/models.py`):**
   - Convoluções com restauração de resolução e ativação não-negativa Softplus garantindo densidades físicas válidas $D(x, y) \ge 0$.

### C. Cadernos Didáticos Documentados Célula a Célula
1. **Notebook 01 ([`01_pre_transformacao_alinhamento.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/01_pre_transformacao_alinhamento.ipynb)):**
   - Executa exatamente o mesmo protocolo óptico e radiométrico consolidado no projeto:
     - Desdistorção grande-angular Brown-Conrady no sensor RAW (ADR 001).
     - Casamento de campo visual (FOV matching) proporcional a 0.58 (ADR 005).
     - Equalização adaptativa local CLAHE no espaço Lab térmico (`clipLimit=3.0`).
     - Alinhamento geométrico subpixel de solo via matriz de homografia (ADR 007 / ADR 008).
   - Gera o contrato padrão em `output/01_pre_transformacao/`.
2. **Notebook 02 ([`02_contagem_pessoas_rgbtcc.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/02_contagem_pessoas_rgbtcc.ipynb)):**
   - Detecção inteligente de hardware (GPU CUDA nativa ou CPU multi-threaded).
   - Ingestão exclusiva do contrato de insumos (princípio de Separação de Preocupações).
   - Normalização estatística rigorosa do benchmark RGBT-CC:
     - $\text{RGB}: \mu=[0.407, 0.389, 0.396], \sigma=[0.241, 0.246, 0.242]$
     - $\text{Térmica}: \mu=[0.492, 0.168, 0.430], \sigma=[0.317, 0.174, 0.191]$
   - Execução da inferência na rede neural `LiuzywenRGBTCCNet`.
   - Geração de heatmaps JET sobrepostos, ROI com zoom em pedestres e gravação de telemetria MLOps.

---

## 2. Validação Experimental e Resultados

Os dois notebooks foram executados ponta a ponta com o interpretador Python do ambiente virtual `.venv`.

### Telemetria de Inferência (`telemetria_contagem.json`)
- **Arquitetura:** `LiuzywenRGBTCCNet (PVTv2 + MSTTrans + MSDTrans)`
- **Total de Parâmetros:** 36.911.842 (~147.6 MB em FP32)
- **Resolução de Inferência:** $640 \times 512$ px (múltiplo exato de 32)
- **Resolução do Contrato:** $1280 \times 1024$ px
- **Latência:** ~701 ms em CPU multi-threaded (1.43 FPS)
- **Contagem Total Integrada:** 3.768,79 (estimativa contínua de densidade)
- **Token Coarse Count:** 0.87

### Artefatos de Saída Gerados
| Estágio | Artefato | Descrição |
| :--- | :--- | :--- |
| **01_pre_transformacao** | `rgb_preprocessed.jpg` | Imagem óptica retificada e alinhada (693 KB) |
| **01_pre_transformacao** | `thermal_preprocessed.jpg` | Imagem térmica com CLAHE e alinhamento afim (548 KB) |
| **01_pre_transformacao** | `blend_alta_precisao.jpg` | Blend 50/50 de auditoria de sobreposição (605 KB) |
| **01_pre_transformacao** | `painel_alinhamento_multimodal.jpg` | Painel consolidado 2x2 com zoom no solo (1.05 MB) |
| **01_pre_transformacao** | `metadata_preprocessing.json` | Registro de parâmetros de calibração |
| **02_contagem** | `density_map.npy` | Matriz NumPy contínua bidimensional (5.24 MB) |
| **02_contagem** | `heatmap_sobre_rgb.jpg` | Heatmap JET sobreposto no canal visual (622 KB) |
| **02_contagem** | `heatmap_sobre_termica.jpg` | Heatmap JET sobreposto no canal térmico (467 KB) |
| **02_contagem** | `zoom_roi_pedestres_densidade.jpg` | Recorte ampliado com zoom nos pedestres (75 KB) |
| **02_contagem** | `painel_contagem_multimodal.jpg` | Painel 2x2 consolidado de contagem (1.31 MB) |
| **02_contagem** | `telemetria_contagem.json` | Métricas de execução estruturadas em JSON |
