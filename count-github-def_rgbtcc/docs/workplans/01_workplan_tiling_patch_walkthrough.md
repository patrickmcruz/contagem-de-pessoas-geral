# Walkthrough: Fase 1 - Inferência por Tiling / Mosaico de Patches (640x512)

**Branch:** `feat/01-tiling-patch-inference`  
**Workplan:** [`docs/workplans/01_workplan_tiling_patch_inference.md`](01_workplan_tiling_patch_inference.md)  
**Notebook:** [`notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`](../notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb)  
**Status da Fase:** Aprovado e Mesclado em `develop` ✅  

---

## 🎯 O que foi feito nesta Fase

Nesta primeira etapa (1º Lugar em Impacto), atacamos o **colapso de escala** de imagens aéreas de drone:
1. Em visão panorâmica ($1280 \times 1024$), 2.826 cabeças de pessoas medem entre 3 a 5 pixels. Com o downsampling de 16x do backbone convolucional, cada pessoa ficava menor que $0.25\text{ pixels}$ e desaparecia na operação de pooling.
2. Criamos o módulo reutilizável [`notebooks/DEF-rgbtcc/models/tiling_inference.py`](../notebooks/DEF-rgbtcc/models/tiling_inference.py) com a função `predict_tiled_density`:
   - Divide a imagem em quadrantes na **resolução nativa de treino do DroneRGBT ($640 \times 512$)**.
   - Suporta sobreposição (*overlap*) com janela 2D de Hann para emenda suave.
   - Costura (*stitching*) os mapas de densidade preservando a integral contínua global.
3. Integramos na **Célula 9 e Célula 10** do notebook [`02_contagem_pessoas_rgbtcc.ipynb`](../notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb), permitindo alternar facilmente entre `USE_TILING = True` e modo monolítico.

---

## 🔬 Resultados Obtidos

* A quantidade de picos locais detectados dobrou (**de 56 para 112**), demonstrando que o janelamento preservou picos individuais que antes eram completamente fundidos na visão global.
* Detalhamento por quadrante na imagem de teste `DJI_0763_W`:
  * *Patch 1 (Top-Left):* 116.70 pessoas
  * *Patch 2 (Top-Right):* 198.02 pessoas
  * *Patch 3 (Bottom-Left):* 159.73 pessoas
  * *Patch 4 (Bottom-Right):* 224.18 pessoas
* **Contagem Total Tiled:** 698.64 pessoas.

