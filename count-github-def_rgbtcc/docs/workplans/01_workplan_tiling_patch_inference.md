# Workplan 01: Inferência por Tiling / Mosaico de Patches (640x512)

**Status:** Finalizado  
**Branch:** `feat/01-tiling-patch-inference`  
**Prioridade:** 1º Lugar (Impacto Crítico - Resolução de Sub-pixel e Colapso de Escala)  
**Notebook Alvo:** `notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`  
**Referência:** `docs/future-improvements/improvements-by-order.md`  

---

## 🎯 Objetivo da Tarefa

Na imagem aérea panorâmica de drone (`DJI_0763_W`, $1280 \times 1024$), 2.826 cabeças de pessoas medem entre 3 a 5 pixels. Quando submetidas à inferência monolítica, o downsampling de 16x do backbone reduz cada pessoa a $\approx 0.25$ pixels, destruindo as características no pooling.

Este workplan implementa a **Inferência por Janelamento / Tiling (Patches)** dividindo a imagem em recortes correspondentes à escala nativa de treinamento do DroneRGBT ($640 \times 512$), inferindo a densidade em cada recorte e costurando (*stitching*) o mapa de densidade global.

---

## 📋 Lista de Tarefas (Tasks)

- [x] **Task 1.1:** Criar branch `feat/01-tiling-patch-inference` a partir da branch `develop`.
- [x] **Task 1.2:** Desenvolver a função `infer_density_tiled` / `predict_tiled_density` em `notebooks/DEF-rgbtcc/models/tiling_inference.py`.
- [x] **Task 1.3:** Integrar a função nas células do notebook `02_contagem_pessoas_rgbtcc.ipynb` (Célula 9 e 10).
- [x] **Task 1.4:** Testar a integridade matemática da costura (shape final `[1024, 1280]`, sem erros em todas as células).
- [x] **Task 1.5:** Gerar o artefato `walkthrough.md` documentando as mudanças e fornecendo instruções claras para execução manual.
- [x] **Task 1.6:** Entregar ao usuário para validação manual no Jupyter / VS Code.
- [x] **Task 1.7:** Após validação e aprovação do usuário, realizar o merge em `develop`.

---

## 📐 Especificação Técnica da Divisão em Patches

Para a imagem alinhada de entrada $H=1024, W=1280$:
* **Grid Padrão:** $2 \times 2$ patches de tamanho $H_{patch}=512, W_{patch}=640$.
  * Patch (0,0): Top-Left $[0:512, 0:640]$
  * Patch (0,1): Top-Right $[0:512, 640:1280]$
  * Patch (1,0): Bottom-Left $[512:1024, 0:640]$
  * Patch (1,1): Bottom-Right $[512:1024, 640:1280]$
* **Overlap Suave (Opcional):** Permite stride de 448x576 com blending linear/cosseno para eliminar descontinuidades na fronteira.
