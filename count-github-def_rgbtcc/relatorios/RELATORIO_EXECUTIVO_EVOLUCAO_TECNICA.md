Aqui está a explicação rápida e direta de cada um dos 3 pilares da base tecnológica do **Novo Algoritmo**:

## O QUE FOI FEITO PARA MELHORIA CONTÍNUA DA APLICAÇÃO DO MODELO

---

### 1. 🧩 Mosaico Nativo (*Tiling Patch Inference*)
* **O que faz:** Em vez de encolher a foto aérea inteira (o que fazia as pessoas virarem apenas 1 ou 2 pixels borrados e sumirem da contagem), o sistema divide a imagem em quadrantes de alta resolução ($640 \times 512$ pixels com sobreposição suave).
* **Impacto no Negócio:** Acaba com a "cegueira de escala" em fotos de drones. A IA consegue enxergar detalhes individuais da cabeça e ombros de cada pessoa, mesmo em multidões densas e distantes.

---

### 2. 📐 Bayesian Abs (*Integração Bayesiana Absoluta*)
* **O que faz:** O modelo foi treinado com uma formulação matemática chamada *Bayesian Loss*, que prevê a probabilidade de presença de pessoas em cada ponto do espaço. Substituímos divisores artificiais antigos pela integração direta e contínua dos valores absolutos ($\text{torch.abs}(\text{density})$).
* **Impacto no Negócio:** Elimina distorções e números artificiais "mágicos". A densidade de calor e luz agora é convertida com fidelidade matemática direta no número real de pessoas, sem subestimar a massa do público.

---

### 3. 🧹 Supressão Inteligente de Fundo (*Background Noise Cutoff*)
* **O que faz:** Drones capturam grandes áreas de asfalto quente, calçadas, telhados e copas de árvores que emitem calor residual difuso. Esse filtro aplica uma margem de corte nos cantos e um piso adaptativo dinâmico (cortando pequenos ruídos abaixo de 5% da intensidade máxima da cena).
* **Impacto no Negócio:** Remove os falsos-positivos térmicos. O sistema garante que só seja contabilizado o calor concentrado com formato corporal humano, impedindo que o calor acumulado no asfalto ou no chão seja somado como "pessoas fantasmas".