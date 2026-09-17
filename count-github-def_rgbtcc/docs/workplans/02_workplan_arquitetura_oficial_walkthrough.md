# Walkthrough: Fase 2 - Integração da Arquitetura Oficial do Artigo (`dm.py`)

**Branch Ativa:** `feat/02-integracao-arquitetura-oficial`  
**Workplan:** [`docs/workplans/02_workplan_arquitetura_oficial_dm.md`](file:///home/patrickcruz/Git/projects/contagem-de-pessoas/count-github-def_rgbtcc/docs/workplans/02_workplan_arquitetura_oficial_dm.md)  
**Notebook:** [`notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`](file:///home/patrickcruz/Git/projects/contagem-de-pessoas/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb)  
**Status da Fase:** Pronto para Validação Manual do Usuário 🚀  

---

## 🎯 O que foi feito nesta Fase

Nesta segunda etapa (2º Lugar em Impacto), substituímos o modelo convolucional reduzido (0.76M parâmetros) pela **arquitetura neural oficial completa do paper** (*A Dual-Modulation Framework for RGB-T Crowd Counting*, arXiv:2509.17079, Feng et al., 2025):
1. **Módulo Oficial Portado:** Criamos [`notebooks/DEF-rgbtcc/models/dm_official.py`](file:///home/patrickcruz/Git/projects/contagem-de-pessoas/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/models/dm_official.py) contendo a implementação autêntica do autor (`Net`, `SpatiallyModulatedAttention`, `AdaptiveCrossModalFusion`, `TransformerEncoder`, `reg()`).
2. **Backbone VGG-19 ImageNet:** Carregamento automático dos pesos pré-treinados do VGG-19 já cacheados localmente.
3. **Escala Paramétrica:** O modelo saltou de **0.76M para 34.14M parâmetros**, representando toda a capacidade representacional multimodal proposta no artigo.
4. **Fábrica Dinâmica:** Atualizamos [`models/def_rgbtcc_net.py`](file:///home/patrickcruz/Git/projects/contagem-de-pessoas/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/models/def_rgbtcc_net.py) com seletor transparente (`architecture="official_dm"` vs `"dual_stream"`).
5. **Notebook Atualizado:** Células 5 e 6 do [`02_contagem_pessoas_rgbtcc.ipynb`](file:///home/patrickcruz/Git/projects/contagem-de-pessoas/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb) agora possuem o seletor `ARCHITECTURE = "official_dm"` e telemetria de parâmetros.

---

## 🔬 Resultados Comparativos Obtidos na Execução

Com a mesma cena panorâmica de drone (`DJI_0763_W`, Ground Truth: 2.826 pessoas):

| Métrica | Base Anterior (Toy CNN) | Fase 1 (Tiling + Toy CNN) | Fase 2 (Tiling + Arquitetura Oficial) | Evolução |
| :--- | :---: | :---: | :---: | :---: |
| **Parâmetros** | 0.76 M | 0.76 M | **34.14 M** | +44x capacidade |
| **Contagem Predita** | ~91 ou 684 | 698.64 | **1.356.16 pessoas** | **+94% de aproximação** |
| **Pessoas Reais (GT)** | 2.826 | 2.826 | **2.826** | Alvo fixo |
| **Erro Normalizado (NAE)** | 75.8% | 75.3% | **52.0%** | **Redução de 23.3 pontos** |
| **RMSE de Contagem** | 2.142 | 2.127 | **1.469** | **Redução de 658 erros** |
| **Tempo de Inferência** | 0.05 s | 0.185 s | **0.191 s (5.2 FPS)** | Quase em tempo real |
| **Ponderação AFM** | 50% / 50% | 50% / 50% | **58.3% RGB / 41.7% Térmica** | Dinâmica real |

---

## 👨‍💻 Instruções para sua Execução Manual

1. **Abrir o notebook:**
   No VS Code, abra [`notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`](file:///home/patrickcruz/Git/projects/contagem-de-pessoas/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb).
2. **Reiniciar o Kernel:**
   Selecione o kernel do ambiente virtual: `./notebooks/.venv/bin/python`.
   Clique em **"Restart Kernel and Run All Cells"**.
3. **Inspecionar as Células:**
   - **Célula 6:** Confirme a mensagem:
     `[✓] Modelo 'Oficial DEF-rgbtcc (arXiv:2509.17079)' Carregado com Sucesso! (34.14 Milhões de parâmetros)`.
   - **Célula 10:** Observe a nova contagem por quadrantes e o peso AFM calculado.
   - **Célula 12 e 22:** Veja a estimativa saltando para **1.356 pessoas** (vs 2.826 do Ground Truth).
4. **Validar e Responder:**
   Assim que você executar e validar, envie **"Etapa aprovada!"** no chat.
   Com a sua aprovação:
   - Faremos o merge da branch `feat/02-integracao-arquitetura-oficial` na `develop`.
   - Iniciaremos a **Fase 3: Eliminação do Fator 0.0001 e Integração Bayesiana Abs**.
