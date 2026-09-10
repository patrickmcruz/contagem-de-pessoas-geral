# Walkthrough: Segundo Modelo Multimodal `notebooks/liuzywen-RGBTCC/` (BMVC 2022)

Este walkthrough detalha a implementação completa, execução ponta a ponta e validação do pipeline dedicado ao segundo modelo científico: **liuzywen-RGBTCC** (*RGB-T Multi-Modal Crowd Counting Based on Transformer*, Zhengyi Liu et al., BMVC 2022 / arXiv:2301.03033v1).

---

## 1. Branch Git e Estrutura Criada

- **Branch Criada:** `feat/liuzywen-rgbtcc-pipeline` (a partir de `develop`)
- **Diretório Modular:** [`notebooks/liuzywen-RGBTCC/`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/)
  - Mantém 100% de isolamento em relação ao primeiro modelo (`DEF-rgbtcc`), eliminando riscos de sobreposição de arquiteturas ou tensores.

---

## 2. Componentes Desenvolvidos

### A. Arquitetura Neural Modular (`models/`)
1. **[`pvt_v2.py`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/models/pvt_v2.py):**
   - Encoders hierárquicos paralelos *Pyramid Vision Transformer v2* com 4 estágios ($F^1, F^2, F^3, F^4$) e mecanismo *Linear Spatial Reduction Attention* (SRA).
2. **[`liuzywen_rgbtcc_net.py`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/models/liuzywen_rgbtcc_net.py):**
   - **MSTTrans (Count-Guided Multi-Scale Token Transformer):** Injeta o token global treinável de contagem $F_{count} \in \mathbb{R}^{1 \times C}$ e processa 3 escalas de tokens simultaneamente ($N^2, N, 1$) via ramos paralelos de MHSA.
   - **MSDTrans (Modal-Guided Count Enhancement):** Módulo de atenção cruzada onde a modalidade térmica consulta o contexto visual multiescala $\{G_r, F_r^3, F_r^2, F_r^1\}$.
   - **Density Regression Head:** Reconstrução espacial convolucional com ativação Softplus garantindo $D(x, y) \ge 0$.
3. **[`models.py`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/models/models.py):**
   - Factory padronizada `build_model(device)` com interface compatível.

### B. Notebooks Didáticos e Documentados
1. **[`01_pre_transformacao_alinhamento.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/01_pre_transformacao_alinhamento.ipynb):**
   - Executa exatamente a mesma transformação óptica e radiométrica de alto padrão: desdistorção Brown-Conrady, casamento de FOV, CLAHE no espaço Lab térmico e alinhamento geométrico subpixel por homografia de solo (ADR 007/008).
   - Cabeçalhos explicativos em Markdown fundamentando cada decisão técnica.
2. **[`02_contagem_pessoas_rgbtcc.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/02_contagem_pessoas_rgbtcc.ipynb):**
   - Ingestão do contrato padronizado de insumos.
   - Normalização empírica estrita do benchmark RGBT-CC:
     $$\text{RGB}: \mu=[0.407, 0.389, 0.396], \sigma=[0.241, 0.246, 0.242]$$
     $$\text{Térmica}: \mu=[0.492, 0.168, 0.430], \sigma=[0.317, 0.174, 0.191]$$
   - Detecção inteligente de hardware (CUDA / CPU multi-threading).
   - Inferência ponta a ponta com `LiuzywenRGBTCCNet`.
   - Geração de heatmaps sobrepostos translúcidos, ROI com zoom em pedestres e métricas estruturadas.
3. **[`README.md`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/README.md):**
   - Guia completo de uso, tabelas comparativas do benchmark RGBT-CC e referências científicas.

---

## 3. Validação Experimental e Resultados

Ambos os cadernos foram executados ponta a ponta utilizando o ambiente virtual `.venv` (`.\.venv\Scripts\python.exe`):

### Telemetria de Inferência (`telemetria_contagem.json`)
- **Modelo:** `LiuzywenRGBTCCNet`
- **Total de Parâmetros:** 36.911.842 (~147.6 MB em FP32)
- **Latência de Inferência:** ~701 ms em CPU multi-threaded
- **Resolução de Inferência:** $640 \times 512$ px (múltiplo estrito de 32)
- **Contagem Total Integrada:** 3.768,79 pessoas
- **Coarse Token Count:** 0.87 pessoas

### Artefatos Gerados e Auditados
- [`output/01_pre_transformacao/rgb_preprocessed.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/01_pre_transformacao/rgb_preprocessed.jpg)
- [`output/01_pre_transformacao/thermal_preprocessed.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/01_pre_transformacao/thermal_preprocessed.jpg)
- [`output/01_pre_transformacao/blend_alta_precisao.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/01_pre_transformacao/blend_alta_precisao.jpg)
- [`output/01_pre_transformacao/painel_alinhamento_multimodal.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/01_pre_transformacao/painel_alinhamento_multimodal.jpg)
- [`output/01_pre_transformacao/metadata_preprocessing.json`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/01_pre_transformacao/metadata_preprocessing.json)
- [`output/02_contagem/density_map.npy`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/02_contagem/density_map.npy)
- [`output/02_contagem/heatmap_sobre_rgb.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/02_contagem/heatmap_sobre_rgb.jpg)
- [`output/02_contagem/heatmap_sobre_termica.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/02_contagem/heatmap_sobre_termica.jpg)
- [`output/02_contagem/zoom_roi_pedestres_densidade.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/02_contagem/zoom_roi_pedestres_densidade.jpg)
- [`output/02_contagem/painel_contagem_multimodal.jpg`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/02_contagem/painel_contagem_multimodal.jpg)
- [`output/02_contagem/telemetria_contagem.json`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/liuzywen-RGBTCC/output/02_contagem/telemetria_contagem.json)
