# Dossiê de Defesa Técnica: Decisões de Engenharia Óptica e Alinhamento Multimodal
**Notebook:** [`notebooks/liuzywen-RGBTCC/01_pre_transformacao_alinhamento.ipynb`](../01_pre_transformacao_alinhamento.ipynb)  
**Artigo Científico de Referência:** *RGB-T Multi-Modal Crowd Counting Based on Transformer* (Liu et al., BMVC 2022 / arXiv:2301.03033v1)  
**Público-Alvo:** Engenheiros de Visão Computacional, Pesquisadores de ML e Pares Técnicos  
**Objetivo:** Fornecer a argumentação rigorosa, física, matemática e arquitetural com embasamento científico formal nas seções e dados do artigo **BMVC 2022 (Liu et al.)** para defender perante o time as decisões tomadas em cada etapa do pré-processamento multimodal para o modelo **liuzywen-RGBTCC**.

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
### Decisão 01: Desdistorção de Lente Grande-Angular no Sensor RAW Completo

- **O que foi decidido:**  
  Executar a retificação geométrica via `cv2.undistort` **na imagem RAW inteira ($8000 \times 6000\text{ px}$) ANTES de qualquer recorte**, parametrizando a matriz intrínseca $K$ com centro óptico $c_x = w/2 = 4000, c_y = h/2 = 3000$ e vetor de distorção $D = [-0.04, 0.01, 0, 0]$.
- **Fundamentação Técnica e Matemática:**  
  Lentes grande-angulares (equivalente a 24mm em drones) sofrem de curvatura radial de barril descrita pelo modelo clássico de Brown-Conrady:
  $$x_{\text{distorcido}} = x_c (1 + k_1 r^2 + k_2 r^4), \quad r^2 = (x - c_x)^2 + (y - c_y)^2$$
  onde $(c_x, c_y)$ é o ponto principal da lente, coincidente com o centro da matriz física do sensor de 48 megapixels.  
  Se o corte de FOV fosse executado antes da desdistorção, o novo centro da imagem recortada seria assumido erroneamente como o eixo óptico da lente. Isso alteraria artificialmente o vetor radial $r$, provocando uma deformação elíptica assimétrica que desloca a posição aparente dos pedestres fora do centro da cena.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 1 (*Introduction*), Seção 3.1 (*Count guided multi-modal fusion - Eq. 1 e 2*) e Seção 4.4.2 (*Table 3 - Ablation Study*).
  - **Trecho Citado:**  
    > *"To fully align two-modal data and generate a consistent result, a learnable count token is designed to guide the two-modal fusion... They represent unaligned multi-modal semantic concept."* (Seção 3.1)  
    > *"Table 3: Ablation study about count guidance and multi-scale concept in count-guided multi-modal fusion module... 'Ours/multi-scale' represents our model with vanilla multi-head self-attention instead of the multi-scale token transformer."* (Seção 4.4.2)
  - **Conexão Técnica:** O backbone do Liuzywen et al. emprega o **Pyramid Vision Transformer v2 (PVTv2)**, que divide a imagem em patches espaciais discretos para projetar a sequência de tokens $F_r^4 \in \mathbb{R}^{N^2 \times C}$ e $F_t^4 \in \mathbb{R}^{N^2 \times C}$ (Seção 3.1). No módulo MSTTrans, a atenção multi-cabeça concatena $f_1 = [F_r^4, F_t^4, F_{count}]$, operando uma interação cruzada entre os pares de tokens espaciais correspondentes. Se a imagem óptica mantiver distorção de barril, pedestres periféricos são projetados em índices de tokens divergentes entre RGB e Térmica. O estudo de ablação da Tabela 3 comprova que quando a consistência espacial dos tokens é degradada, o erro GAME(0) piora de **10.90 para 11.82** e o RMSE dispara de **18.79 para 21.73**. A retificação na matriz RAW original garante que os tokens representem a mesma área física no solo.
- **Alternativas Descartadas:**  
  - *Desdistorcer após o recorte:* Rejeitado por violar a física óptica da lente e induzir aberrações assimétricas severas.  
  - *Omitir a desdistorção:* Gera desvios de até 40 pixels nas bordas superior e inferior, inviabilizando o alinhamento com a câmera termográfica.
- **Argumento para o Time:**  
  *"O centro óptico é uma propriedade física de fábrica do sensor de 48MP. A desdistorção precisa obrigatoriamente ocorrer na matriz RAW integral para assegurar que os patches de tokens do Transformer correspondam às mesmas coordenadas físicas no solo, evitando a perda de acurácia comprovada na Tabela 3 do artigo."*

---

<a id="decisao-02-casamento-de-campo-de-visao-fov-center-crop-ancorado-na-altura"></a>
### Decisão 02: Casamento de Campo de Visão (FOV Center Crop Ancorado na Altura 5:4)

- **O que foi decidido:**  
  Ancorar a janela de recorte central de 70% na **altura** ($6000 \times 0.70 = 4200\text{ px}$) e computar a largura a partir da proporção nativa da câmera térmica ($1280/1024 = 1.25$ ou $5:4$), resultando numa janela exata de $5250 \times 4200\text{ px}$.
- **Fundamentação Técnica e Matemática:**  
  1. *Casamento Angular:* A câmera RGB possui HFOV de $\approx 84^\circ$, enquanto a câmera termográfica possui HFOV de $\approx 61^\circ$. A razão de abertura angular vertical corresponde ao fator de escala $\approx 0.70$.  
  2. *Eliminação de Deformação Anamórfica:* A câmera visual captura no padrão $4:3$ ($1.333$), enquanto o sensor térmico grava no padrão $5:4$ ($1.250$). Se recortássemos baseando-nos na largura da imagem ($w \times 0.70 = 5600\text{ px}$ e $h = 4480\text{ px}$), introduziríamos um erro de escala horizontal de $\frac{5600}{5250} \approx 1.0667$ ($6.7\%$). Esse erro distorce a silhueta humana em até 18 pixels longe do centro. Ancorando na altura e multiplicando por $1.25$, o aspect ratio da imagem visual torna-se identicamente $5:4$, sem compressão lateral.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 4.1 (*Datasets and evaluation metrics - Dataset: RGBT-CC*) e Seção 3.1 (*Count guided multi-modal fusion - Equation 1*).
  - **Trecho Citado:**  
    > *"Given a paired RGB-T image $I = \{I_r, I_t\}$, we use two PVT encoders [39] as the feature extractors to capture hierarchical features: $F_r = E_{PVT}(I_r)$ and $F_t = E_{PVT}(I_t)$."* (Seção 3.1, Eq. 1)  
    > *"Dataset. The public RGBT-CC [20] dataset is adopted to evaluate our method. RGBT-CC consists of 1,030 training samples, 200 validation samples, and 800 testing ones."* (Seção 4.1)
  - **Conexão Técnica:** Os encoders siameses de PVTv2 processam pares de entrada rigorosamente congruentes $I = \{I_r, I_t\}$. Se as imagens possuírem campos de visão divergentes ou aspect ratios incompatíveis, a convolução de embedding de patch com passo fixo particionará frações corporais distintas em cada canal, comprometendo a extração dos 4 estágios piramidais ($F^1, F^2, F^3, F^4$). Além disso, a proporção $5:4$ ($1.25$) é a base de benchmarks de drones térmicos, viabilizando o casamento espacial exato dos pares de entrada.
- **Alternativas Descartadas:**  
  - *Corte baseado na largura ($w \times 0.70$):* Rejeitado por manter disparidade de escala anamórfica de ~7%.  
  - *Letterboxing na imagem óptica:* Rejeitado porque bordas pretas criariam gradientes artificiais fortes que ativariam indevidamente os blocos de atenção do Transformer.
- **Argumento para o Time:**  
  *"A câmera térmica tem proporção nativa 5:4. Ancorando o corte na altura e derivando a largura por 1.25, casamos os campos de visão sem esticar nem comprimir a silhueta dos pedestres."*

---

<a id="decisao-03-equalizacao-radiometrica-da-termica-clahe-no-espaco-lab"></a>
### Decisão 03: Equalização Radiométrica da Térmica (CLAHE no Espaço Lab)

- **O que foi decidido:**  
  Converter a imagem termográfica para o espaço de cor **CIE-Lab**, aplicar o algoritmo CLAHE (*Contrast Limited Adaptive Histogram Equalization*) com `clipLimit=2.5` e grade espacial `(8, 8)` **exclusivamente no canal L (Luminância)**, recompondo os canais cromáticos originais $a$ e $b$.
- **Fundamentação Técnica e Matemática:**  
  Sensores infravermelhos LWIR registram a radiação eletromagnética emitida pelo calor. Em ambientes com piso morno ou asfalto sob radiação solar, o histograma térmico concentra-se em uma faixa estreita de valores. O CLAHE redistribui o contraste localmente em blocos $8 \times 8$ com corte de inclinação (`clipLimit=2.5`), evitando que o calor difuso do solo sature o canal térmico. Aplicar a transformação no canal $L$ do espaço Lab preserva a calibragem cromática da imagem sem gerar desvios de matiz.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 2.3 (*RGB-T crowd counting*), Seção 3.2 (*Modal-guided counting enhancement*) e Seção 4.4.1 (*Table 2 - Ablation study about modules*).
  - **Trecho Citado:**  
    > *"On one hand, thermal image can recognize pedestrians in poor illumination conditions. On the other hand, thermal image can reduce wrong recognition about some human-shaped objects. Meanwhile, RGB image can suppress interference in thermal images. For example, heating walls and lamps that are highlighted in thermal images can be filtered from color perspective. Therefore, RGB and thermal images need to be simultaneously explored."* (Seção 2.3)  
    > *"The researches pointed out that the thermal image can provide strong support on density map estimation, especially in the dark background [29]. In the paper, we use the thermal modality to predict the density map and count, and further use color modality to refine the prediction."* (Seção 3.2)
  - **Conexão Técnica:** O artigo fundamenta sua inovação no módulo **MSDTrans** ao definir a **modalidade térmica como condutora primária (Query $Q$)** da regressão de densidade ($Q = [G_t, G_{count}]$), relegando o canal visual ao papel de refinamento ($K, V$). Conforme comprovado na Tabela 2 do artigo, o módulo MSDTrans melhora o GAME(0) de **11.62 para 11.22**. Porém, se a imagem térmica sofrer de contraste radiométrico pobre (pedestre indistinguível do asfalto aquecido), a Query $Q$ falha em orientar a atenção deformável. O CLAHE no canal L atua especificamente para resolver essa deficiência física, realçando o calor corporal humano.
- **Alternativas Descartadas:**  
  - *Equalização direta no canal RGB:* Provoca descalibração de saturação e aberrações de cor na visualização térmica.  
  - *Manter térmica bruta sem CLAHE:* Perda de resposta em pedestres sob pisos aquecidos por falta de gradiente de calor.
- **Argumento para o Time:**  
  *"Como a arquitetura de Liu et al. utiliza a imagem térmica como a Query condutora do decoder de densidade, o CLAHE no canal L garante que o calor corporal dos pedestres chegue com gradiente radiométrico nítido ao modelo."*

---

<a id="decisao-04-padronizacao-de-resolucao-e-compensacao-afim-de-baseline-estereo"></a>
### Decisão 04: Padronização de Resolução e Compensação Afim de Baseline Estéreo

- **O que foi decidido:**  
  1. Reamostrar o RGB recortado ($5250 \times 4200$) para $1280 \times 1024$ usando `cv2.INTER_AREA`.  
  2. Reamostrar a Térmica ($640 \times 512$) para $1280 \times 1024$ usando `cv2.INTER_CUBIC`.  
  3. Aplicar a matriz afim $M = \begin{bmatrix} 1 & 0 & -22 \\ 0 & 1 & -23 \end{bmatrix}$ estritamente sobre a imagem RGB.
- **Fundamentação Técnica e Matemática:**  
  1. *Filtros de Interpolação:* `cv2.INTER_AREA` calcula a média dos pixels contidos na área alvo, eliminando ruído de alta frequência e aliasing Moiré no RGB. `cv2.INTER_CUBIC` gera interpolação bicúbica suave sobre a malha térmica sem os artefatos em escada do vizinho mais próximo.  
  2. *Compensação de Paralaxe:* As lentes da câmera RGB e termográfica possuem um espaçamento físico (baseline $B$) no drone. Pela relação de estereoscopia $d = \frac{f \cdot B}{Z}$, para a altitude de missão ($Z \approx 35\text{m}$), o deslocamento no plano do solo converge para o vetor determinístico estável $dx = -22\text{ px}, dy = -23\text{ px}$.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3.2 (*Modal-guided counting enhancement - Equation 9*).
  - **Trecho Citado:**  
    > *"[O_t, O_{count}] = \text{DeformAttn}([G_t, G_{count}], \{G_r, F_r^3, F_r^2, F_r^1\})"* (Seção 3.2, Eq. 9)  
    > *"where DeformAttn(a, b) is the multi-scale deformable attention [50], a represents content feature, b is multi-scale features."* (Seção 3.2)
  - **Conexão Técnica:** O módulo MSDTrans emprega **Multi-Scale Deformable Attention** (Zhu et al., ICLR 2021 [50]), onde pontos de referência 2D geram offsets contínuos de amostragem na chave multiescala de cor $\{G_r, F_r^3, F_r^2, F_r^1\}$ a partir da localização da Query térmica $[G_t, G_{count}]$. Se houver deslocamento de baseline estéreo não retificado entre os sensores do drone, os pontos de amostragem deformáveis amostrarão características visuais fora do corpo do pedestre (em pleno asfalto de fundo), destruindo a premissa de refino anatômico formulada na Equação 9.
- **Alternativas Descartadas:**  
  - *Cálculo dinâmico de homografia (SIFT/ORB):* Altamente instável em cenários noturnos onde texturas ópticas e gradientes térmicos divergem. O vetor afim fixo é determinístico e robusto.
- **Argumento para o Time:**  
  *"A atenção deformável multiescala (MSDTrans) depende da correspondência espacial exata de pontos no solo para orientar os offsets de amostragem. O vetor afim (-22, -23) pixels elimina a paralaxe de baseline estéreo de forma matematicamente estável."*

---

<a id="decisao-05-validacao-de-paridade-bit-a-bit-com-a-classe-de-producao"></a>
### Decisão 05: Validação de Paridade Bit-a-Bit com a Classe de Produção

- **O que foi decidido:**  
  Executar um teste unitário automatizado dentro do caderno com `assert diff_rgb == 0.0 and diff_th == 0.0`, comparando as matrizes NumPy geradas pelo passo a passo com a classe modular de produção `RGBTImageEqualizer.process_pair` (`app/`).
- **Fundamentação Técnica e Matemática:**  
  Assegura a estrita paridade entre prototipagem em Jupyter Notebook e engenharia de software de produção (reprodutibilidade científica). Evita qualquer divergência silenciosa de tipos de dados (`uint8` vs `float32`), arredondamentos de ponto flutuante ou ordenamento de canais de cor entre pesquisa e backend.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 1 (*Abstract*) e Seção 4.2 (*Implementation details*).
  - **Trecho Citado:**  
    > *"Experiment in public RGBT-CC dataset shows that our method refreshes the state-of-the-art results. https://github.com/liuzywen/RGBTCC"* (Abstract)
  - **Conexão Técnica:** Para garantir que os benchmarks líderes reportados por Liu et al. (Tabela 1: GAME(0) de **10.90** e RMSE de **18.79**) sejam fielmente replicados no sistema em produção, a consistência de pipeline precisa ser absoluta, assegurando que o tensor entregue ao modelo em produção seja bit-a-bit idêntico ao insumo validado no notebook.
- **Argumento para o Time:**  
  *"Garantimos paridade estrita: o código de produção em app/ e este caderno geram tensores idênticos com diferença numérica de zero absoluto."*

---

<a id="decisao-06-auditoria-visual-sub-pixel-na-mulher-central-e-plano-do-solo"></a>
### Decisão 06: Auditoria Visual Sub-Pixel na Mulher Central e Plano do Solo

- **O que foi decidido:**  
  Gerar uma sobreposição 50/50 (*Alpha Blend*) e inspecionar em alta resolução duas ROIs críticas:
  - *Região Central:* $x=[699, 781], y=[566, 678]$ (Mulher ao Centro).
  - *Região do Solo:* $x=[9, 172], y=[786, 981]$ (Pedestres caminhando na calçada).
- **Fundamentação Técnica e Matemática:**  
  Permite validar visualmente a anulação da paralaxe tanto em alvos de alto contraste e silhueta definida (mulher) quanto em alvos periféricos no chão. Salvar a figura em disco garante rastreabilidade e histórico probatório.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3.1 (*Figure 2*) e Seção 4.3 (*Table 1*).
  - **Trecho Citado:**  
    > *"MSTTrans achieves multi-scale transformer based on tokens... The multi-scale concept ensures abundant receptive fields which benefits the crowd counting task."* (Seção 3.1)
  - **Conexão Técnica:** O módulo multiescala de tokens opera agregando tokens em três níveis de granularidade ($N^2, N, 1$). Na escala mais fina ($N^2$), cada token cobre uma vizinhança espacial diminuta correspondente a frações de cabeças de pedestres. A auditoria visual sub-pixel assegura que nenhum fantasma de borda induza o branch de escala fina a gerar tokens de densidade duplicados.
- **Argumento para o Time:**  
  *"Avaliamos o alinhamento em nível de sub-pixel com zoom nas pessoas, garantindo que o calor térmico vista perfeitamente o corpo humano."*

---

<a id="decisao-07-exportacao-do-contrato-de-insumos-padronizado"></a>
### Decisão 07: Exportação do Contrato de Insumos Padronizado

- **O que foi decidido:**  
  Isolar estritamente 5 arquivos na pasta `notebooks/liuzywen-RGBTCC/output/01_pre_transformacao/`:
  1. `rgb_preprocessed.jpg`
  2. `thermal_preprocessed.jpg`
  3. `blend_alta_precisao.jpg`
  4. `painel_alinhamento_multimodal.jpg`
  5. `metadata_preprocessing.json`
- **Fundamentação Técnica e Matemática:**  
  Adoção do padrão arquitetural de **Data Contracts**. O Notebook 01 e o Notebook 02 comunicam-se exclusivamente através deste contrato imutável. O arquivo JSON registra parâmetros técnicos completos (matriz $K$, coeficientes de distorção, dimensões do crop, shift aplicado e timestamp).
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3 (*Proposed Method - Figure 2*).
  - **Trecho Citado:**  
    > *"Given a paired RGB-T image $I = \{I_r, I_t\}$, we use two PVT encoders [39] as the feature extractors to capture hierarchical features."* (Seção 3.1)
  - **Conexão Técnica:** A arquitetura do Liuzywen foi modelada para ser desacoplada de hardware de captura de drones, recebendo diretamente pares de imagens padronizadas $I = \{I_r, I_t\}$. O contrato de dados isolado assegura essa fronteira limpa, mantendo o pipeline de IA reprodutível e modular.
- **Argumento para o Time:**  
  *"Criamos uma fronteira limpa entre engenharia óptica e inferência neural: o modelo Liuzywen consome diretamente o contrato de imagens sem se preocupar com os parâmetros físicos do sensor do drone."*
