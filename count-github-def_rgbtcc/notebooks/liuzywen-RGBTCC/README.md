# Pipeline Multimodal RGBTCC (Liu et al., BMVC 2022)

Este diretório contém a implementação completa, desacoplada e reprodutível do modelo de contagem de pessoas multimodal apresentado no artigo:

> **"RGB-T Multi-Modal Crowd Counting Based on Transformer"**  
> Zhengyi Liu, Wei Wu, Yacheng Tan, Guanghui Zhang (Anhui University)  
> *British Machine Vision Conference (BMVC 2022) / arXiv:2301.03033v1*  
> Repositório de Referência: [github.com/liuzywen/RGBTCC](https://github.com/liuzywen/RGBTCC)  
> Artigo em PDF: [`docs/2301.03033v1.pdf`](docs/2301.03033v1.pdf)

---

## 1. Visão Geral da Arquitetura

O modelo baseia-se em um paradigma de **Regressão de Mapa de Densidade Espacial 2D** (*Density Map Estimation*) multimodal guiado por Transformers:

```
                          ┌───────────────────────────┐
                          │   PAR MULTIMODAL ALINHADO │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       [Stream 1: Óptico RGB]                       [Stream 2: Térmico LWIR]
       • Backbone: PVTv2-B3                         • Backbone: PVTv2-B3
       • 4 Estágios Hierárquicos                    • 4 Estágios Hierárquicos
       • F_r = {F_r^1, F_r^2, F_r^3, F_r^4}         • F_t = {F_t^1, F_t^2, F_t^3, F_t^4}
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                   [MSTTrans: Count-Guided Multi-Scale Token Transformer]
                   • Token global de contagem F_count ∈ R^{1 x C}
                   • Fusão em 3 escalas: Initial (N^2), Middle (N), Large (1)
                   • Multi-Head Self-Attention (MHSA) + Restauração FC + MLP
                   • Saída: G = [G_r, G_t, G_count]
                                        ▼
                   [MSDTrans: Modal-Guided Counting Enhancement]
                   • Query (Q): [G_t, G_count] (Térmica guiada pelo token)
                   • Key/Value (K, V): {G_r, F_r^3, F_r^2, F_r^1} (Multi-escala visual)
                   • Multi-Scale Deformable Cross-Attention: [O_t, O_count]
                                        ▼
                       [Regression Head & Density Map]
                       • Convoluções 3x3 + 1x1 + Ativação Softplus
                       • Contagem Total: P_pred = ∑ D(x, y)
```

---

## 2. Estrutura Organizada do Diretório

```
notebooks/liuzywen-RGBTCC/
├── 01_pre_transformacao_alinhamento.ipynb # Estágio 1: Transformação, Calibração e Alinhamento
├── 02_contagem_pessoas_rgbtcc.ipynb       # Estágio 2: Inferência com Transformer (PVTv2 + MSTTrans)
├── README.md                              # Esta documentação
│
├── docs/
│   └── 2301.03033v1.pdf                   # Artigo científico oficial do BMVC 2022
│
├── input/                                 # Insumos Brutos (RAW)
│   ├── DJI_0789_W.JPG                     # Imagem óptica grande-angular (24mm)
│   └── DJI_0790_T.JPG                     # Imagem termográfica LWIR
│
├── models/                                # Arquitetura Neural Modular
│   ├── __init__.py                        # Exportações do pacote
│   ├── pvt_v2.py                          # Backbone Pyramid Vision Transformer v2
│   ├── liuzywen_rgbtcc_net.py             # MSTTrans, MSDTrans e Regression Head
│   └── models.py                          # Factory build_model unificada
│
└── output/                                # Saídas Estruturadas por Estágio
    ├── 01_pre_transformacao/              # Contrato de Insumos Gerado no Estágio 1
    │   ├── rgb_preprocessed.jpg           # RGB retificado e alinhado (1280x1024)
    │   ├── thermal_preprocessed.jpg       # Térmica com realce CLAHE e alinhada (1280x1024)
    │   ├── blend_alta_precisao.jpg        # Blend 50/50 para validação visual
    │   ├── painel_alinhamento_multimodal.jpg
    │   └── metadata_preprocessing.json
    │
    └── 02_contagem/                       # Relatórios de Inferência Gerados no Estágio 2
        ├── painel_contagem_multimodal.jpg # Painel consolidado 2x2 com heatmaps e ROI
        ├── zoom_roi_pedestres_densidade.jpg
        ├── heatmap_sobre_rgb.jpg
        ├── heatmap_sobre_termica.jpg
        ├── density_map.npy                # Matriz NumPy contínua do mapa de densidade
        └── telemetria_contagem.json       # Métricas de latência, hardware e contagem
```

---

## 3. Como Executar o Pipeline

### Passo 1: Executar o Notebook 01
Abra e execute [`01_pre_transformacao_alinhamento.ipynb`](01_pre_transformacao_alinhamento.ipynb).  
Ele processa os pares brutos de `input/` aplicando calibração de lente grande-angular, casamento de FOV, CLAHE radiométrico e alinhamento geométrico afim/mesh warp, salvando o contrato padronizado em `output/01_pre_transformacao/`.

### Passo 2: Executar o Notebook 02
Abra e execute [`02_contagem_pessoas_rgbtcc.ipynb`](02_contagem_pessoas_rgbtcc.ipynb).  
Ele carrega o contrato de insumos, aplica a normalização estrita do RGBT-CC, instancia a rede `LiuzywenRGBTCCNet` no melhor acelerador de hardware disponível (GPU CUDA ou CPU multi-threaded) e gera os mapas de densidade, visualizações e telemetria em `output/02_contagem/`.

---

## 4. Desempenho no Benchmark RGBT-CC (Liu et al., BMVC 2022)

Resultados publicados no dataset público **RGBT-CC** (1.030 treino, 200 validação, 800 teste):

| Modelo | Fonte | GAME(0) ↓ | GAME(1) ↓ | GAME(2) ↓ | GAME(3) ↓ | RMSE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CSRNet | CVPR 2018 | 20.40 | 23.58 | 28.03 | 35.51 | 35.26 |
| BL | ICCV 2019 | 18.70 | 22.55 | 26.83 | 34.62 | 32.67 |
| DM-Count | NeurIPS 2020 | 16.54 | 20.73 | 25.23 | 32.23 | 27.22 |
| CMCRL | CVPR 2021 | 15.61 | 19.95 | 24.69 | 32.89 | 28.18 |
| TAFNet | ISCAS 2022 | 12.38 | 16.98 | 21.86 | 30.19 | 22.45 |
| DEFNet | TITS 2022 | 11.90 | 16.08 | 20.19 | 27.27 | 21.09 |
| **liuzywen-RGBTCC (Ours)** | **BMVC 2022** | **10.90** | **14.81** | **19.02** | **26.14** | **18.79** |
