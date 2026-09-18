# Workplan 04: Supressão de Fundo e Limiarização de Ruído (ε)

**Status:** Em Desenvolvimento  
**Branch:** `feat/04-supressao-fundo-ruido`  
**Prioridade:** 4º Lugar (Limpeza de Ruído Residual em Áreas Vazias)  
**Notebook Alvo:** `notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`  
**Referência:** `docs/future-improvements/improvements-by-order.md` | Background Modeling (Feng et al., 2025)  

---

## 🎯 Objetivo da Tarefa

Em tomadas aéreas de drone, mais de 70% da área da imagem é composta por regiões vazias (asfalto, grama, árvores, telhados). Como ativações convolucionais residuais possuem valores ínfimos porém positivos ($\epsilon \approx 0.0001$ a $0.0003$), a soma sobre 1.310.720 pixels acumula um ruído de fundo que infla a contagem ou cria falsas detecções em áreas desabitadas.

Além disso, as bordas extremas da imagem sofrem com reflexões de convolução (*padding border artifacts*), gerando picos espúrios nas fronteiras externas.

Este workplan implementa:
1. **Supressão de Borda:** Neutralização de artefatos de convolução nas margens externas (8 pixels).
2. **Limiarização de Piso de Ruído ($\epsilon$):** Eliminação de valores difusos abaixo do limiar de piso ($5\%$ da amplitude de pico do mapa de densidade).
3. **Telemetria de Fundo:** Quantificação exata de pessoas espúrias neutralizadas e percentual da imagem com fundo zerado.

---

## 📋 Lista de Tarefas (Tasks)

- [x] **Task 4.1:** Criar branch `feat/04-supressao-fundo-ruido` a partir de `develop`.
- [x] **Task 4.2:** Implementar a função `suppress_background_noise` em `notebooks/DEF-rgbtcc/models/tiling_inference.py`.
- [x] **Task 4.3:** Integrar a supressão de fundo nas Células 9 e 10 de `02_contagem_pessoas_rgbtcc.ipynb`, expondo parâmetros configuráveis (`ENABLE_BG_SUPPRESSION`, `BORDER_MARGIN`, `BG_CUTOFF_RATIO`).
- [x] **Task 4.4:** Atualizar a Célula 12 e as projeções visuais (heatmaps) para refletir o mapa de densidade limpo.
- [x] **Task 4.5:** Testar a inferência com e sem supressão de fundo em script automatizado (GPU CUDA RTX 4090).
- [x] **Task 4.6:** Atualizar o `walkthrough.md` com os novos resultados e instruções de execução.
- [x] **Task 4.7:** Entregar ao usuário para validação manual no Jupyter / VS Code (Walkthrough: [`04_workplan_supressao_fundo_ruido_walkthrough.md`](04_workplan_supressao_fundo_ruido_walkthrough.md)).
- [x] **Task 4.8:** Após validação e aprovação do usuário, realizar o merge em `develop`.
