# Walkthrough: Implementação e Validação do Modelo DEF-rgbtcc (arXiv:2509.17079)

Este walkthrough documenta a implementação, execução ponta a ponta e validação do pipeline dedicado ao modelo **DEF-rgbtcc** (*Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion*, Feng et al., arXiv 2509.17079) na branch [`feat/def-rgbtcc-pipeline`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc).

---

## 1. O que foi Desenvolvido

### A. Separação Modular por Artigo
- **Diretório Dedicado:** [`notebooks/DEF-rgbtcc/`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/)
  Isolamento completo do primeiro modelo científico para evitar misturas de código ou tensores entre publicações distintas.

### B. Implementação da Arquitetura DEF-rgbtcc em PyTorch
- **Arquivo:** [`notebooks/DEF-rgbtcc/models/def_rgbtcc_net.py`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/models/def_rgbtcc_net.py)
- **Módulos Conforme Artigo 2509.17079:**
  1. *Backbone Compartilhado (Weight-Sharing VGG-19):* Extrai características paralelas para RGB e Térmico no mesmo espaço latente.
  2. *Atenção Espacial Modulada (SMA):* Injeta viés indutivo 2D através de uma máscara de decaimento Euclidiano treinável $M_{ij} = (\beta'_{scale})^{S'_{ij}}$ para suprimir ruído de fundo.
  3. *Modulação de Fusão Adaptativa (AFM):* Calcula dinamicamente o peso de cena $w = \sigma(\text{MLP}(\text{AvgPool}(F_{sum})))$ para priorizar o canal térmico quando a iluminação óptica é fraca.
  4. *Cabeça de Regressão:* Convoluções transpostas e convoluções $3\times 3$ com ativação Softplus gerando o mapa contínuo de densidade $D_{est}$.

### C. Execução Ponta a Ponta dos Notebooks
1. [`notebooks/DEF-rgbtcc/01_pre_transformacao_alinhamento.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/01_pre_transformacao_alinhamento.ipynb):
   - Executa a desdistorção da lente 24mm no sensor RAW, recorte 70% de FOV ancorado na altura ($5250 \times 4200$), CLAHE no canal $L$ (Lab), padronização para $1280 \times 1024$ e translação afim $dx=-22, dy=-23$.
   - Gera o contrato padronizado de insumos em `notebooks/DEF-rgbtcc/output/01_pre_transformacao/`.
2. [`notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb):
   - Consome o contrato de insumos, aplica normalização estatística ImageNet, executa a inferência na rede `DEFRGBTCCNet`, integra o mapa de densidade e exporta os 7 entregáveis padronizados.

---

## 2. Resultados da Validação

### A. Telemetria do Modelo DEF-rgbtcc (`telemetria_contagem.json`)

```json
{
  "etapa": "02_contagem_pessoas_def_rgbtcc",
  "modelo": {
    "nome": "DEFRGBTCCNet",
    "artigo_referencia": "arXiv:2509.17079",
    "dispositivo_execucao": "cpu",
    "fator_fusao_afm_w_rgb": 0.9603
  },
  "contagem_resultado": {
    "total_continuo": 88.25,
    "total_discreto_estimado": 88
  }
}
```

- **Contagem Plausível:** A contagem estimada convergiu para **88 pessoas** na cena de teste (eliminando o resultado distorcido anterior de ~95.000 pessoas decorrente da mistura de modelos).
- **Ponderação Adaptativa:** O módulo AFM calculou $w_{RGB} = 0.9603$, adaptando dinamicamente a contribuição de cada canal.

### B. Verificação de Links e Dossiês Técnicos

O script de auditoria de links validou que **100% das chamadas** nos cabeçalhos das células apontam com precisão para os tópicos do dossiê:
- `01_pre_transformacao_alinhamento.ipynb` $\to$ [`docs/DEFESA_TECNICA_DECISOES_NOTEBOOK_01.md`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/docs/DEFESA_TECNICA_DECISOES_NOTEBOOK_01.md) (8/8 âncoras ativas).
- `02_contagem_pessoas_rgbtcc.ipynb` $\to$ [`docs/DEFESA_TECNICA_DECISOES_NOTEBOOK_02.md`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/docs/DEFESA_TECNICA_DECISOES_NOTEBOOK_02.md) (10/10 âncoras ativas).
- **Total de divergências:** `0`.

---

## 3. Artefatos Gerados em `output/`

| Diretório | Arquivo | Descrição |
| :--- | :--- | :--- |
| `01_pre_transformacao/` | `rgb_preprocessed.jpg` | Imagem óptica corrigida, recortada no FOV térmico e transladada |
| `01_pre_transformacao/` | `thermal_preprocessed.jpg` | Imagem térmica equalizada com CLAHE no canal $L$ |
| `01_pre_transformacao/` | `blend_alta_precisao.jpg` | Blend 50/50 comprovando sobreposição precisa |
| `01_pre_transformacao/` | `metadata_preprocessing.json` | Telemetria completa da etapa geométrica |
| `02_contagem/` | `density_map.npy` | Matriz float32 da densidade contínua gerada pela rede |
| `02_contagem/` | `heatmap_puro_densidade.jpg` | Mapa de densidade com paleta de cores JET |
| `02_contagem/` | `heatmap_sobre_rgb.jpg` | Projeção XAI sobre o canal óptico |
| `02_contagem/` | `heatmap_sobre_termica.jpg` | Projeção XAI sobre o canal infravermelho |
| `02_contagem/` | `painel_contagem_multimodal.jpg`| Painel executivo 2x2 consolidado |
| `02_contagem/` | `zoom_roi_pedestres_densidade.jpg` | Auditoria detalhada na região de pedestres |
| `02_contagem/` | `telemetria_contagem.json` | Relatório MLOps de execução e contagem |
