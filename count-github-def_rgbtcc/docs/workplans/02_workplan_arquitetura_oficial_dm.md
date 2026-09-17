# Workplan 02: Integração da Arquitetura Oficial do Artigo (dm.py)

**Status:** Em Desenvolvimento  
**Branch:** `feat/02-integracao-arquitetura-oficial`  
**Prioridade:** 2º Lugar (Base Arquitetural do Modelo - VGG-19 + SMA + AFM)  
**Notebook Alvo:** `notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`  
**Referência:** `docs/future-improvements/improvements-by-order.md` | Paper arXiv:2509.17079  

---

## 🎯 Objetivo da Tarefa

No código original, a rede em execução era uma CNN leve de 0.76M parâmetros (`DualStreamRGBTNet`), treinada com poucas épocas em cenas pequenas (`calibrated_count: 58.0`).

Este workplan porta a **arquitetura neural oficial completa** do paper (*A Dual-Modulation Framework for RGB-T Crowd Counting*, Feng et al., 2025), localizada no repositório oficial recém-atrelado (`/home/patrickcruz/Git/projects/RGBT-Crowd-Counting/models/dm.py`):
1. **Backbone VGG-19 compartilhado** (26M de parâmetros com pesos pré-treinados do ImageNet já em cache).
2. **Transformers SMA (Spatially Modulated Attention)** com máscara de decaimento espacial $M$.
3. **Módulo AFM (Adaptive Cross-Modal Fusion)** para fusão multimodal adaptativa.
4. **Head de Regressão Bayesiana** com ativação `torch.abs(density)`.

---

## 📋 Lista de Tarefas (Tasks)

- [x] **Task 2.1:** Criar branch `feat/02-integracao-arquitetura-oficial` a partir de `develop`.
- [x] **Task 2.2:** Portar o modelo oficial para `notebooks/DEF-rgbtcc/models/dm_official.py` preservando todas as classes originais do autor.
- [x] **Task 2.3:** Criar `OfficialDMWrapper` e atualizar a factory `build_def_rgbtcc_model` em `models/def_rgbtcc_net.py` para permitir seleção dinâmica (`model_type="official_dm"` vs `"dual_stream"`).
- [x] **Task 2.4:** Ajustar o tratamento de escala do mapa de densidade em `models/tiling_inference.py` para suportar nativamente a saída 1/8 do modelo oficial.
- [x] **Task 2.5:** Atualizar as Células 5 e 6 do notebook `02_contagem_pessoas_rgbtcc.ipynb` com seletor de arquitetura e telemetria de parâmetros.
- [x] **Task 2.6:** Testar a inferência de ponta a ponta em script automatizado (GPU CUDA RTX 4090).
- [x] **Task 2.7:** Atualizar o `walkthrough.md` com instruções detalhadas para o usuário.
- [ ] **Task 2.8:** Entregar ao usuário para execução manual no Jupyter / VS Code.
- [ ] **Task 2.9:** Após validação e aprovação do usuário, realizar o merge em `develop`.
