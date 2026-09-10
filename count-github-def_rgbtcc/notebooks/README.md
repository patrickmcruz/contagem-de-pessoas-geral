# Pipeline Didático de Cadernos Multimodais (RGBT)

Este diretório contém o pipeline modular e desacoplado em **dois Jupyter Notebooks sequenciais** para calibração óptica, equalização radiométrica e contagem de pessoas via Visão Computacional profunda.

---

## Estrutura Organizada do Diretório

```
notebooks/
├── 01_pre_transformacao_alinhamento.ipynb             # Estágio 1: Transformação, Calibração e Alinhamento Óptico
├── 02_contagem_pessoas_rgbtcc.ipynb                   # Estágio 2: Inferência Multimodal, Densidade e Contagem
├── 03_estudo_somente_corte_vs_pipeline_completo.ipynb # Estudo Comparativo: Somente Corte vs Pipeline Completo
│
├── input/                                   # ENTRADAS EXCLUSIVAMENTE BRUTAS (RAW)
│   ├── DJI_0789_W.JPG                       # Imagem óptica RGB (Lente grande-angular 24mm)
│   ├── DJI_0790_T.JPG                       # Imagem termográfica LWIR (Infravermelho)
│   └── _arquivo_experimentos/               # Arquivamento de recortes e testes manuais legados
│
└── output/                                  # SAÍDAS ORGANIZADAS 1:1 COM OS NOTEBOOKS
    ├── 01_pre_transformacao/                # 🔹 GERADO PELO NOTEBOOK 01 (Contrato de Insumo)
    │   ├── rgb_preprocessed.jpg             # RGB retificado e alinhado (1280x1024)
    │   ├── thermal_preprocessed.jpg         # Térmica equalizada com CLAHE (1280x1024)
    │   ├── blend_alta_precisao.jpg          # Blend 50/50 de sobreposição para verificação
    │   ├── painel_alinhamento_multimodal.jpg# Painel visual com auditoria de pedestres
    │   └── metadata_preprocessing.json      # Registro dos parâmetros técnicos aplicados
    │
    ├── 02_contagem/                         # 🔹 GERADO PELO NOTEBOOK 02 (Relatórios e Inferência)
    │   ├── painel_contagem_multimodal.jpg   # Painel consolidado 2x2 com heatmaps
    │   ├── zoom_roi_pedestres_densidade.jpg # Zoom em pedestres validando densidade da IA
    │   ├── heatmap_sobre_rgb.jpg            # Projeção de calor sobre a imagem óptica
    │   ├── heatmap_sobre_termica.jpg        # Projeção de calor sobre a imagem térmica
    │   ├── density_map.npy                  # Matriz NumPy contínua do mapa de densidade
    │   └── telemetria_contagem.json         # Métricas de latência e contagem final de pessoas
    │
    ├── 03_estudo_corte_vs_pipeline/         # 🔹 GERADO PELO NOTEBOOK 03 (Estudo Comparativo)
    │   ├── comparativo_somente_corte_zoom_mulher.jpg   # Auditoria comparativa na mulher central
    │   ├── comparativo_somente_corte_zoom_pedestre.jpg # Auditoria comparativa nos pedestres no solo
    │   ├── comparativo_auditoria_mulher_corrigida.jpg  # Diagnóstico antes/depois da calibração
    │   └── resumo_estudo_comparativo.json              # Resumo técnico estruturado do estudo
    │
    └── _arquivo_experimentos/               # Histórico arquivado de testes legados (01_raw a 07_test_01)
```

---

## 🚀 Como Executar o Pipeline

1. **Abra o Notebook 01:** [`01_pre_transformacao_alinhamento.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/01_pre_transformacao_alinhamento.ipynb)
   - Execute todas as células.
   - O notebook lê as fotos brutas de `input/`, aplica as correções de lente, CLAHE e alinhamento afim, e grava tudo em `output/01_pre_transformacao/`.

2. **Abra o Notebook 02:** [`02_contagem_pessoas_rgbtcc.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/02_contagem_pessoas_rgbtcc.ipynb)
   - Execute todas as células.
   - O notebook consome automaticamente o contrato de `output/01_pre_transformacao/`, seleciona o hardware adaptativo (RTX 4090, RTX 3080, RTX 5070 ou CPU), executa a inferência na rede neural `ThermalRGBNet`, gera os mapas de calor sobrepostos e calcula a contagem final de pessoas, gravando os artefatos em `output/02_contagem/`.

---

## 💻 Suporte Multi-GPU Inteligente

O Notebook 02 possui detecção adaptativa de hardware (`detect_compute_device`), garantindo portabilidade entre diferentes máquinas:

| Ambiente / Placa | Arquitetura | Comportamento do Notebook |
| :--- | :--- | :--- |
| **NVIDIA GeForce RTX 4090 / 4080** | Ada Lovelace (`sm_89`) | **GPU CUDA Nativo**: Execução direta com aceleração máxima por hardware e Tensor Cores. |
| **NVIDIA GeForce RTX 3090 / 3080** | Ampere (`sm_86`) | **GPU CUDA Nativo**: Execução direta acelerada por hardware. |
| **NVIDIA GeForce RTX 5070 / 5080** | Blackwell (`sm_120`) | **Compatibilidade Segura**: Detecta se os binários locais do PyTorch já compilam `sm_120`; caso contrário, executa transparentemente em CPU sem erros de kernel. |
| **Ambientes CPU / Servidor** | x86_64 / ARM | **Multi-threading CPU**: Executa a inferência em menos de 2 segundos sem requerer GPU dedicada. |

---

## 🔬 Exploração Livre de Novos Métodos

Graças ao desacoplamento por **Contrato de Dados** (`output/01_pre_transformacao/`):
- **Novo método de alinhamento ou calibração?** Trabalhe exclusivamente no **Notebook 01**. O Notebook 02 consumirá os novos dados sem alteração de código.
- **Novo modelo de IA ou arquitetura de contagem?** Trabalhe exclusivamente no **Notebook 02**. Não é necessário reprocessar as imagens brutas da câmera.
