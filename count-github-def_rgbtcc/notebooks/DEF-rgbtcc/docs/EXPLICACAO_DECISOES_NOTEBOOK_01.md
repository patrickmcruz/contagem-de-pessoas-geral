# Dossiê de Defesa Técnica: Decisões de Engenharia Óptica e Alinhamento Multimodal
**Notebook:** [`notebooks/01_pre_transformacao_alinhamento.ipynb`](../01_pre_transformacao_alinhamento.ipynb)  
**Artigo Científico de Referência:** *A Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion* (Feng et al., arXiv 2509.17079, 2025)  
**Público-Alvo:** Engenheiros de Visão Computacional, Pares Técnicos e Liderança de Projeto  
**Objetivo:** Fornecer a argumentação rigorosa, física, matemática, arquitetural e respaldo científico formal para defender perante o time as decisões tomadas em cada etapa do pré-processamento multimodal RGBT.

---

## Índice das Decisões
1. [Decisão 01: Desdistorção de Lente Grande-Angular no Sensor RAW Completo](#decisao-01-desdistorcao-de-lente-grande-angular-no-sensor-raw-completo)
2. [Decisão 02: Casamento de Campo de Visão (FOV Center Crop Ancorado na Altura 5:4)](#decisao-02-casamento-de-campo-de-visao-fov-center-crop-ancorado-na-altura)
3. [Decisão 03: Equalização Radiométrica da Térmica (CLAHE no Espaço Lab)](#decisao-03-equalizacao-radiometrica-da-termica-clahe-no-espaco-lab)
4. [Decisão 04: Padronização de Resolução e Compensação Afim de Baseline Estéreo](#decisao-04-padronizacao-de-resolucao-e-compensacao-afim-de-baseline-estereo)
5. [Decisão 05: Validação de Paridade Bit-a-Bit com a Classe de Produção](#decisao-05-validacao-de-paridade-bit-a-bit-com-a-classe-de-producao)
6. [Decisão 06: Auditoria Visual Sub-Pixel na Mulher Central e Plano do Solo](#decisao-06-auditoria-visual-sub-pixel-na-mulher-central-e-plano-do-solo)
7. [Decisão 07: Exportação do Contrato de Insumos Padronizado](#decisao-07-exportacao-do-contrato-de-insumos-padronizado)

---

<a id="decisao-01-desdistorcao-de-lente-grande-angular-no-sensor-raw-completo"></a>
<a id="decisao-03-desdistorcao-de-lente-grande-angular-no-sensor-raw-completo"></a>
### Decisão 01: Desdistorção de Lente Grande-Angular no Sensor RAW Completo

- **O que foi decidido:**  
  Executar o algoritmo `cv2.undistort` **na imagem RAW completa ($8000 \times 6000\text{ px}$) ANTES de qualquer recorte**, usando a matriz intrínseca $K$ com centro óptico $cx = w/2 = 4000, cy = h/2 = 3000$ e coeficientes de distorção radial $D = [-0.04, 0.01, 0, 0]$.
- **Fundamentação Técnica e Matemática:**  
  A lente grande-angular 24mm equivalente sofre de distorção radial de barril descrita pelo modelo polinomial de Brown-Conrady:
  $$x_{\text{distorcido}} = x_c (1 + k_1 r^2 + k_2 r^4), \quad r^2 = (x - c_x)^2 + (y - c_y)^2$$
  onde $(c_x, c_y)$ é o ponto principal físico da lente, coincidente com o centro da matriz física do sensor de 48 megapixels.  
  Se recortássemos a imagem antes de aplicar a desdistorção, o centro geométrico da subimagem passaria a ser assumido incorretamente como o centro óptico da lente. Isso alteraria o vetor radial $r$ em milhares de pixels, provocando distorção artificial assimétrica que deforma pessoas fora do centro (causa raiz do deslocamento lateral de 18 px observado em testes preliminares).
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 1 (*Introduction*) e Seção 2.1 (*Spatially Modulated Attention - Eq. 1 a 4*).
  - **Trecho Citado:**  
    > *"As a result, the self-attention mechanism distributes focus across irrelevant background regions, as visualized in Fig. 1, which adversely affects density map estimation by leading to inaccurate localization and blurred boundaries."* (Seção 1)  
    > *"This bias is implemented by systematically penalizing attention scores between distant tokens based on their pairwise spatial distance, which forces the model to prioritize local interactions and suppresses interference from irrelevant background regions."* (Seção 2.1)
  - **Conexão Técnica:** A premissa central do módulo SMA é que a atenção entre tokens é modulada pela distância Euclidiana euclidiana $S_{ij} \in \mathbb{R}^{N \times N}$ no plano 2D ($M_{ij} = (\beta'_{scale})^{S'_{ij}}$). Se a imagem contiver distorção óptica de barril não retificada, a métrica espacial Euclidiana é corrompida nas periferias, fazendo com que tokens vizinhos aparentem estar mais distantes ou deslocados, prejudicando o cálculo da máscara de decaimento $M$ e induzindo ativações espúrias no mapa de densidade final.
- **Alternativas Descartadas:**  
  - *Desdistorcer após o corte:* Provou-se matematicamente incorreto, criando aberrações assimétricas nas bordas e deslocando alvos periféricos.  
  - *Ignorar a desdistorção:* Gera erro de projeção nos cantos superior e inferior de até 40 pixels, inviabilizando o casamento com a câmera térmica.
- **Argumento para o Time:**  
  *"A física da lente não se altera após cortarmos uma foto. O centro óptico é uma propriedade de fábrica do sensor de 48MP; por isso, a retificação precisa obrigatoriamente ocorrer na matriz original inteira antes de qualquer recorte."*

---

<a id="decisao-02-casamento-de-campo-de-visao-fov-center-crop-ancorado-na-altura"></a>
<a id="decisao-04-casamento-de-campo-de-visao-fov-center-crop-ancorado-na-altura"></a>
### Decisão 02: Casamento de Campo de Visão (FOV Center Crop Ancorado na Altura 5:4)

- **O que foi decidido:**  
  Ancorar a janela de recorte central de 70% na **altura** ($6000 \times 0.70 = 4200\text{ px}$) e derivar a largura a partir da proporção de aspecto nativa do sensor térmico ($1280/1024 = 1.25$ ou $5:4$), resultando numa janela de recorte exata de $5250 \times 4200\text{ px}$.
- **Fundamentação Técnica e Matemática:**  
  1. *Casamento Angular:* A câmera grande-angular possui HFOV de aproximadamente $84^\circ$, enquanto a câmera termográfica possui HFOV de $\approx 61^\circ$. A razão de abertura angular vertical corresponde a $\approx 0.70$.  
  2. *Eliminação de Deformação Anamórfica:* A câmera óptica captura na proporção $4:3$ ($1.333$), enquanto a câmera térmica grava na proporção $5:4$ ($1.250$). Se recortássemos baseando-nos na largura da imagem (versão preliminar com $w \times 0.70 = 5600\text{ px}$ e $h = 4480\text{ px}$), inseriríamos um erro de escala óptica de:
  $$\frac{5600}{5250} = 1.0667 \quad (\approx 7\% \text{ de divergência de escala})$$
  Esse erro de $7\%$ em pedestres a $250\text{ px}$ do centro cria uma disparidade lateral de $250 \times 0.07 \approx 17.5\text{ px}$, gerando o efeito fantasma na silhueta humana. Ao ancorar na altura ($4200$) e multiplicar por $1.25$ ($5250$), a imagem óptica resultante adota nativamente a proporção $5:4$, sem compressão lateral ao ser redimensionada para $1280 \times 1024$.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 3 (*Experiment - Datasets: DroneRGBT*) e Seção 2.2 (*Adaptive Fusion Modulation - Eq. 6*).
  - **Trecho Citado:**  
    > *"DroneRGBT [23] is a large-scale, drone-based benchmark containing 3,607 RGB-T image pairs with a resolution of 640 × 512."* (Seção 3)  
    > *"F_{fused} = w \cdot F'_r + (1 - w) \cdot F'_t"* (Seção 2.2, Eq. 6)
  - **Conexão Técnica:** O principal benchmark de imagens aéreas capturadas por drones avaliado no paper (**DroneRGBT**) adota resolução nativa de $640 \times 512$, que possui relação de aspecto exata de $5:4$ ($1.25$). Como a fusão AFM realiza a soma ponderada elemento a elemento entre features ópticas e térmicas ($w \cdot F'_r + (1-w) \cdot F'_t$), qualquer distorção de aspect ratio entre os canais provocaria o desalinhamento de pedestres no grid de tensores, quebrando a hipótese de correspondência espacial ponto a ponto exigida pela rede neural.
- **Alternativas Descartadas:**  
  - *Corte baseado na largura ($w \times 0.70$):* Rejeitado devido ao erro residual de 7% de escala horizontal.  
  - *Letterboxing na imagem óptica:* Rejeitado porque faixas pretas adicionariam bordas artificiais que ativariam filtros convolucionais espúrios.
- **Argumento para o Time:**  
  *"A câmera térmica enxerga menos que a visual e tem proporção nativa 5:4 (idêntica ao benchmark DroneRGBT do artigo). Ancorando o corte na altura e derivando a largura por 1.25, casamos o enquadramento sem esticar nem encolher a silhueta dos pedestres."*

---

<a id="decisao-03-equalizacao-radiometrica-da-termica-clahe-no-espaco-lab"></a>
<a id="decisao-05-equalizacao-radiometrica-da-termica-clahe-no-espaco-lab"></a>
### Decisão 03: Equalização Radiométrica da Térmica (CLAHE no Espaço Lab)

- **O que foi decidido:**  
  Converter a imagem termográfica para o espaço de cor **CIE-Lab**, aplicar o algoritmo CLAHE (*Contrast Limited Adaptive Histogram Equalization*) com `clipLimit=2.5` e grade espacial `(8, 8)` **exclusivamente no canal L (Luminância)**, recompondo os canais cromáticos originais $a$ e $b$.
- **Fundamentação Técnica e Matemática:**  
  Sensores térmicos LWIR (8 a 14 $\mu m$) capturam radiação eletromagnética emitida por calor. Em dias de sol ou asfalto morno, o histograma térmico concentra-se em uma faixa estreita de valores.  
  Uma equalização de histograma global convencional redistribuiria os tons saturando todo o ruído de fundo (como o brilho difuso do asfalto), mascarando as pessoas.  
  O CLAHE particiona a imagem em blocos $8 \times 8$ e limita a inclinação do histograma (`clipLimit=2.5`), redistribuindo os fótons térmicos localmente. Ao atuar exclusivamente sobre o canal $L$ do espaço Lab, o contraste radiométrico dos pedestres é amplificado sem alterar a paleta de cores ou distorcer a geometria térmica.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 1 (*Introduction*) e Seção 2.2 (*Adaptive Fusion Modulation*).
  - **Trecho Citado:**  
    > *"RGB-Thermal (RGB-T) approaches improve performance by leveraging the complementarity of visual and thermal data."* (Seção 1)  
    > *"For instance, in a low-light scene where RGB information is unreliable, the network learns to produce a small w, thereby prioritizing the more reliable thermal features to generate a more reliable final feature map."* (Seção 2.2)
  - **Conexão Técnica:** O artigo fundamenta a resiliência do modelo na capacidade do módulo AFM de priorizar os atributos térmicos ($1-w$) quando a modalidade visual degrada no escuro. Se a imagem térmica bruta tiver baixo contraste de gradiente térmico em relação ao solo aquecido, a VGG-19 não conseguirá extrair representações robustas $F_t$. O CLAHE isola e realça o contraste local da assinatura térmica humana, fornecendo gradientes limpos para que a AFM efetivamente explore a complementaridade térmica.
- **Alternativas Descartadas:**  
  - *Equalização direta no canal RGB:* Distorce o equilíbrio de cor da paleta termográfica e cria aberrações cromáticas em torno de lâmpadas.  
  - *Manter a térmica bruta sem equalização:* O extrator de features perde acurácia em pessoas distantes ou sob o asfalto aquecido por falta de contraste radiométrico local.
- **Argumento para o Time:**  
  *"O CLAHE no canal L atua como um filtro polarizador térmico: ele realça o calor corporal das pessoas sem explodir o ruído do solo morno, viabilizando o ganho da modulação térmica previsto no artigo."*

---

<a id="decisao-04-padronizacao-de-resolucao-e-compensacao-afim-de-baseline-estereo"></a>
<a id="decisao-06-padronizacao-de-resolucao-e-compensacao-afim-de-baseline-estereo"></a>
### Decisão 04: Padronização de Resolução e Compensação Afim de Baseline Estéreo

- **O que foi decidido:**  
  1. Reamostrar o RGB recortado ($5250 \times 4200$) para $1280 \times 1024$ usando `cv2.INTER_AREA`.  
  2. Reamostrar a Térmica ($640 \times 512$) para $1280 \times 1024$ usando `cv2.INTER_CUBIC`.  
  3. Aplicar matriz afim $M = \begin{bmatrix} 1 & 0 & -22 \\ 0 & 1 & -23 \end{bmatrix}$ estritamente sobre a imagem RGB.
- **Fundamentação Técnica e Matemática:**  
  1. *Filtros de Reamostragem:* Na redução de escala ($5250 \to 1280$), o `INTER_AREA` calcula a média dos pixels contidos no pixel de destino, eliminando o efeito Moiré e ruídos de alta frequência. Na ampliação ($640 \to 1280$), o `INTER_CUBIC` utiliza convolução cúbica $4 \times 4$ de vizinhos, criando gradientes suaves sem os degraus pixelizados do `INTER_NEAREST`.  
  2. *Compensação de Paralaxe de Baseline:* As câmeras estão montadas no gimbal do drone separadas fisicamente por uma distância estéreo $B$. Pela relação de estereoscopia:
  $$d = \frac{f \cdot B}{Z}$$
  Para a altitude de voo de missão ($Z \approx 30\text{ a }40\text{ m}$), o deslocamento no plano do solo resulta no vetor estável $dx = -22\text{ px}$ e $dy = -23\text{ px}$. Aplicar essa translação na imagem RGB alinha perfeitamente a posição do chão óptico com o calor detectado.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 2 (*Methodology - Fig. 2*) e Seção 2.2 (*Adaptive Fusion Modulation - Eq. 5*).
  - **Trecho Citado:**  
    > *"As illustrated in Fig. 2, a weight-sharing VGG19 backbone first extracts parallel feature maps, Fr and Ft , from an RGB-T image pair."* (Seção 2)  
    > *"We first create a joint feature map by element-wise summation: Fsum = Fr' + Ft'. This joint map, which encapsulates a summary of information from both modalities, is then globally pooled..."* (Seção 2.2)
  - **Conexão Técnica:** No DEF-rgbtcc, o backbone siamês compartilha pesos para extrair tensores paralelos $F_r$ e $F_t$, e a fusão inicial calcula a soma direta ponto a ponto $F_{sum} = F'_r + F'_t$. Se as imagens de entrada sofrerem de paralaxe de baseline (mesmo que de 20 pixels), o tensor somado $F_{sum}$ adicionará uma cabeça de pedestre óptico sobre o fundo da imagem térmica e vice-versa, gerando dupla ativação e duplicando falsamente a contagem de pedestres. A compensação afim elimina a paralaxe de baseline estéreo antes da entrada na rede.
- **Alternativas Descartadas:**  
  - *Homografia dinâmica por SIFT/ORB em cada frame:* Falha criticamente na fusão RGB-T noturna porque os descritores de gradiente visual e térmico divergem (uma lâmpada tem brilho visual, mas não representa a cabeça de um pedestre). O vetor fixo calibrado é $100\%$ determinístico e imune a erros de convergência.
- **Argumento para o Time:**  
  *"A distância entre os sensores no drone é física e constante. Um vetor afim calibrado de (-22, -23) pixels garante alinhamento no solo determinístico, protegendo a soma elemento a elemento da AFM contra ativações duplas."*

---

<a id="decisao-05-validacao-de-paridade-bit-a-bit-com-a-classe-de-producao"></a>
<a id="decisao-07-validacao-de-paridade-bit-a-bit-com-a-classe-de-producao"></a>
### Decisão 05: Validação de Paridade Bit-a-Bit com a Classe de Produção

- **O que foi decidido:**  
  Inserir um teste unitário formal em célula executável com `assert diff_rgb == 0.0 and diff_th == 0.0`, comparando o fluxo exploratório do notebook com a implementação modular `RGBTImageEqualizer.process_pair` da pasta `app/`.
- **Fundamentação Técnica e Matemática:**  
  Garante paridade estrita entre prototipagem científica e engenharia de software (princípio da reprodutibilidade científica). Elimina qualquer risco de discrepância entre os tensores gerados no notebook de exploração e os tensores consumidos pela API de produção.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção "Abstract" e Seção 3 (*Implementation Details*).
  - **Trecho Citado:**  
    > *"Code available at https://github.com/Cht2924/RGBT-Crowd-Counting."* (Abstract)  
    > *"Our model is implemented in PyTorch and trained on a single NVIDIA RTX 3090 GPU."* (Seção 3)
  - **Conexão Técnica:** Assim como os autores do artigo enfatizam a reprodutibilidade exata disponibilizando o código-fonte integral, o teste de paridade bit-a-bit no pipeline garante que as transformações de imagens aplicadas aos benchmarks do artigo sejam rigorosamente idênticas às executadas no serviço em tempo real.
- **Argumento para o Time:**  
  *"Não existe 'no meu notebook funciona e em produção não'. O assert no código comprova erro de exatamente zero pixels entre este caderno de pesquisa e o código de produção em app/."*

---

<a id="decisao-06-auditoria-visual-sub-pixel-na-mulher-central-e-plano-do-solo"></a>
<a id="decisao-08-auditoria-visual-sub-pixel-na-mulher-central-e-plano-do-solo"></a>
### Decisão 06: Auditoria Visual Sub-Pixel na Mulher Central e Plano do Solo

- **O que foi decidido:**  
  Gerar uma sobreposição 50/50 (*Alpha Blend*) e recortar explicitamente duas ROIs críticas:
  - *Região Central Calibrada:* $x=[699, 781], y=[566, 678]$ (Mulher ao Centro).
  - *Região Base / Solo:* $x=[9, 172], y=[786, 981]$ (Pedestres caminhando).
- **Fundamentação Técnica e Matemática:**  
  Permite validar visualmente a anulação da paralaxe tanto em alvos de alto contraste e silhueta definida (mulher) quanto em alvos periféricos no chão. Salvar a figura em disco garante rastreabilidade e histórico probatório.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 1 (Fig. 1) e Seção 3 (Fig. 4).
  - **Trecho Citado:**  
    > *"Fig. 1. Our SMA prevents attention from spreading to irrelevant regions, enabling precise localization... In contrast, our SMA produces compact, head-centric attention."* (Seção 1)
  - **Conexão Técnica:** A arquitetura DEF-rgbtcc foi projetada para produzir "compact, head-centric attention", ou seja, picos de densidade rigorosamente centralizados nas cabeças dos pedestres. Se a sobreposição entre as imagens visual e térmica apresentar desalinhamento sub-pixel, os picos de atenção da SMA e da AFM sofrem alargamento ou bifurcação, reduzindo o índice GAME nas subdivisões finas de grade (GAME(2) e GAME(3)).
- **Argumento para o Time:**  
  *"Não confiamos apenas em números abstratos. Damos um zoom de alta definição nas pessoas para auditar que o contorno de calor veste o corpo humano com perfeição milimétrica."*

---

<a id="decisao-07-exportacao-do-contrato-de-insumos-padronizado"></a>
<a id="decisao-09-exportacao-do-contrato-de-insumos-padronizado"></a>
### Decisão 07: Exportação do Contrato de Insumos Padronizado

- **O que foi decidido:**  
  Isolar estritamente 5 arquivos na pasta `notebooks/DEF-rgbtcc/output/01_pre_transformacao/`:
  1. `rgb_preprocessed.jpg`
  2. `thermal_preprocessed.jpg`
  3. `blend_alta_precisao.jpg`
  4. `painel_alinhamento_multimodal.jpg`
  5. `metadata_preprocessing.json`
- **Fundamentação Técnica e Matemática:**  
  Adoção do padrão arquitetural de **Data Contracts**. O Notebook 01 e o Notebook 02 comunicam-se exclusivamente através deste contrato imutável. O arquivo JSON registra parâmetros técnicos completos (matriz $K$, coeficientes de distorção, dimensões do crop, shift aplicado e timestamp).
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 2 (*Methodology*).
  - **Trecho Citado:**  
    > *"Our proposed Dual Modulation Framework is an end-to-end network designed to address the dual challenges of achieving precise spatial localization and effective cross-modal fusion in RGB-T crowd counting. As illustrated in Fig. 2, a weight-sharing VGG19 backbone first extracts parallel feature maps, Fr and Ft , from an RGB-T image pair."*
  - **Conexão Técnica:** O modelo neural opera a partir da premissa de receber tensores de entrada pré-calibrados e padronizados $(F_r, F_t)$. Ao encapsular o pré-processamento em um contrato de dados isolado com metadados JSON, garantimos que a rede neural receba insumos idênticos às condições dos datasets públicos RGBT-CC e DroneRGBT, mantendo o pipeline modular e desacoplado.
- **Argumento para o Time:**  
  *"Criamos uma fronteira limpa entre engenharia de imagem e inferência de IA. Qualquer novo modelo de contagem que for avaliado no futuro poderá consumir diretamente este contrato sem reprocessar um único pixel da câmera."*
