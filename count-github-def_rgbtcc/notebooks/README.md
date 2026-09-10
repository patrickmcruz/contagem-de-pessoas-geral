# Pipeline Didático de Cadernos Multimodais (RGBT)

Este diretório contém o pipeline modular e desacoplado em **dois Jupyter Notebooks sequenciais** para calibração, alinhamento de imagens óptico-térmicas e contagem de pessoas via Visão Computacional profunda.

---

## Estrutura do Pipeline

```
notebooks/
├── 01_pre_transformacao_alinhamento.ipynb  # Estágio 1: Transformação, Calibração e Alinhamento Óptico
├── 02_contagem_pessoas_rgbtcc.ipynb        # Estágio 2: Inferência Multimodal, Densidade e Contagem
│
├── input/                                  # Imagens brutas (RAW) capturadas pelo drone
│   ├── DJI_0789_W.JPG                      # Foto óptica RGB (Lente grande-angular 24mm)
│   └── DJI_0790_T.JPG                      # Foto termográfica LWIR (Infravermelho de onda longa)
│
├── data/
│   └── processed/                          # CONTRATO DE INTERFACE (Insumo padronizado)
│       ├── rgb_preprocessed.jpg            # RGB retificado e alinhado (1280x1024)
│       ├── thermal_preprocessed.jpg        # Térmica equalizada com CLAHE (1280x1024)
│       ├── blend_alta_precisao.jpg         # Blend 50/50 para verificação visual
│       └── metadata_preprocessing.json     # Registro de parâmetros técnicos da transformação
│
└── output/                                 # Artefatos exportados e relatórios de auditoria
    ├── 01_pre_transformacao/               # Gráficos de alinhamento e auditoria da etapa 1
    └── 08_pipeline_final/                  # Painel de contagem, heatmaps, ROI e telemetria JSON
```

---

## 🚀 Como Executar o Pipeline

1. **Abra o Notebook 01:** [`01_pre_transformacao_alinhamento.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/01_pre_transformacao_alinhamento.ipynb)
   - Execute todas as células.
   - O notebook lê as fotos brutas de `input/`, aplica as correções de lente, CLAHE e alinhamento afim, e grava os insumos em `data/processed/`.

2. **Abra o Notebook 02:** [`02_contagem_pessoas_rgbtcc.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/02_contagem_pessoas_rgbtcc.ipynb)
   - Execute todas as células.
   - O notebook consome os arquivos de `data/processed/`, seleciona dinamicamente o hardware (RTX 4090, RTX 3080, RTX 5070 ou CPU), executa a inferência na rede neural `ThermalRGBNet`, gera os mapas de calor sobrepostos e calcula a contagem final de pessoas.

---

## 💻 Suporte Multi-GPU Inteligente

O Notebook 02 foi projetado com detecção adaptativa de hardware (`detect_compute_device`), garantindo que o mesmo código funcione em qualquer estação de trabalho sem necessidade de alteração:

| Ambiente / Placa | Arquitetura | Comportamento do Notebook |
| :--- | :--- | :--- |
| **NVIDIA GeForce RTX 4090 / 4080** | Ada Lovelace (`sm_89`) | **GPU CUDA Nativo**: Execução direta com máxima aceleração e Tensor Cores. |
| **NVIDIA GeForce RTX 3090 / 3080** | Ampere (`sm_86`) | **GPU CUDA Nativo**: Execução direta acelerada por hardware. |
| **NVIDIA GeForce RTX 5070 / 5080** | Blackwell (`sm_120`) | **Compatibilidade Segura**: Detecta se os binários locais do PyTorch já compilam `sm_120`; caso contrário, executa transparentemente em CPU sem erros de kernel. |
| **Ambientes CPU / Colab / Servidor** | x86_64 / ARM | **Multi-threading CPU**: Executa a inferência em menos de 1 segundo sem requerer GPU. |

---

## 🔬 Como Explorar Novos Métodos

Graças ao desacoplamento por **Contrato de Dados** (`data/processed/`):
- **Quer testar um novo método de alinhamento óptico?** (ex: Homografia por SIFT, ECC, alinhamento elástico): trabalhe exclusivamente no **Notebook 01**. O Notebook 02 continuará funcionando perfeitamente.
- **Quer testar outro modelo de IA para contagem?** (ex: YOLOv8-Crowd, CSRNet, DM-Count): trabalhe exclusivamente no **Notebook 02**. Você não precisa recalcular nem reprocessar as imagens brutas da câmera.

Para mais detalhes técnicos e matemáticos, consulte o guia completo em [`docs/GUIA_PIPELINE_NOTEBOOKS_RGBTCC.md`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/docs/GUIA_PIPELINE_NOTEBOOKS_RGBTCC.md).
