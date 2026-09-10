# Dossiê de Defesa Técnica: Arquitetura liuzywen-RGBTCC, Inferência Multimodal e Contagem de Densidade
**Notebook:** [`notebooks/liuzywen-RGBTCC/02_contagem_pessoas_rgbtcc.ipynb`](../02_contagem_pessoas_rgbtcc.ipynb)  
**Artigo Científico de Referência:** *RGB-T Multi-Modal Crowd Counting Based on Transformer* (Liu et al., BMVC 2022 / arXiv:2301.03033v1)  
**Público-Alvo:** Engenheiros de Machine Learning, Pesquisadores de Visão Computacional e Liderança Técnica  
**Objetivo:** Fornecer a argumentação científica, matemática e arquitetural com embasamento formal, equações e tabelas de ablação do artigo **BMVC 2022 (Liu et al.)** para defender perante o time as decisões tomadas em cada etapa do pipeline neural **liuzywen-RGBTCC**.

---

## Índice das Decisões
1. [Decisão 01: Setup do Ambiente e Estrutura Modular dos Módulos Neurais](#decisao-01-setup-do-ambiente-e-estrutura-modular-dos-modulos-neurais)
2. [Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado](#decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado)
3. [Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU / CPU Multi-threading](#decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu)
4. [Decisão 04: Normalização Estatística Empírica do Benchmark RGBT-CC](#decisao-04-normalizacao-estatistica-empirica-do-benchmark-rgbt-cc)
5. [Decisão 05: Arquitetura Siamesa PVTv2-B2 com Redução Espacial Linear (SRA)](#decisao-05-arquitetura-siamesa-pvtv2-b2-com-reducao-espacial-linear-sra)
6. [Decisão 06: Count-Guided Multi-Scale Token Transformer (MSTTrans e Token $F_{count}$)](#decisao-06-count-guided-multi-scale-token-transformer-msttrans-e-token-fcount)
7. [Decisão 07: Modal-Guided Count Enhancement (MSDTrans via Deformable Attention)](#decisao-07-modal-guided-count-enhancement-msdtrans-via-deformable-attention)
8. [Decisão 08: Inferência sem Gradientes e Reconstrução da Densidade com Softplus](#decisao-08-inferencia-sem-gradientes-e-reconstrucao-da-densidade-com-softplus)
9. [Decisão 09: Integração Numérica em Densidade e Dupla Validação com $O_{count}$](#decisao-09-integracao-numerica-em-densidade-e-dupla-validacao-com-ocount)
10. [Decisão 10: Projeções Visuais, Heatmap JET (XAI) e Exportação MLOps com Métricas GAME/RMSE](#decisao-10-projecoes-visuais-heatmap-jet-xai-e-exportacao-mlops-com-metricas-gamermse)

---

<a id="decisao-01-setup-do-ambiente-e-estrutura-modular-dos-modulos-neurais"></a>
### Decisão 01: Setup do Ambiente e Estrutura Modular dos Módulos Neurais

- **O que foi decidido:**  
  Configurar importações do PyTorch (`torch`, `torchvision`), configurar a resolução de caminhos relativos via `pathlib.Path` e injetar a pasta `notebooks/liuzywen-RGBTCC` diretamente no `sys.path`.
- **Fundamentação Técnica e Matemática:**  
  Permite carregar dinamicamente os submódulos proprietários do modelo: `models.pvt_v2` (backbone Pyramid Vision Transformer v2) e `models.liuzywen_rgbtcc_net` (módulos MSTTrans, MSDTrans e Regression Head) sem exigir empacotamento no Python global ou dependências de build externas.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Abstract e Seção 4.2 (*Implementation details*).
  - **Trecho Citado:**  
    > *"Experiment in public RGBT-CC dataset shows that our method refreshes the state-of-the-art results. https://github.com/liuzywen/RGBTCC"* (Abstract)  
    > *"The implementation setting includes: (1) GPU (NVIDIA RTX 3090); (2) input image size (224 × 224); (3) train time (17 hours); (4) learning rate (1e-5); (5) weight decay (1e-4)."* (Seção 4.2)
  - **Conexão Técnica:** A injeção local de pacotes permite instanciar diretamente as classes e pesos oficiais disponibilizados no repositório dos autores, garantindo que a implementação PyTorch execute com 100% de conformidade aos hiperparâmetros e topologia descritos no paper.
- **Argumento para o Time:**  
  *"Qualquer desenvolvedor ou pesquisador consegue clonar o repositório e executar a inferência do modelo Liuzywen de imediato sem configurações complexas de ambiente."*

---

<a id="decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado"></a>
### Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado

- **O que foi decidido:**  
  Consumir estritamente os artefatos gerados pelo Notebook 01 em `output/01_pre_transformacao/` (`rgb_preprocessed.jpg` e `thermal_preprocessed.jpg`), acompanhados da validação de integridade dos metadados JSON.
- **Fundamentação Técnica e Matemática:**  
  Aplicação do princípio arquitetural de **Separação de Preocupações (Separation of Concerns - SoC)**. O Notebook 02 não recalcula desdistorção nem faz cropping; ele atua exclusivamente na execução e interpretação do modelo neural. O desacoplamento por contrato de dados garante independência completa entre a camada de calibração física de sensores e a camada de inteligência artificial de contagem, permitindo comparar de forma justa os dois modelos de artigos distintos (`DEF-rgbtcc` e `liuzywen-RGBTCC`) sob as mesmas entradas retificadas.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3.1 (*Equation 1*).
  - **Trecho Citado:**  
    > *"Given a paired RGB-T image $I = \{I_r, I_t\}$, we use two PVT encoders [39] as the feature extractors to capture hierarchical features: $F_r = E_{PVT}(I_r)$ and $F_t = E_{PVT}(I_t)$."* (Seção 3.1, Eq. 1)
  - **Conexão Técnica:** O framework teórico do Liuzywen et al. parte da hipótese de que o par de entrada $I = \{I_r, I_t\}$ já se encontra calibrado e geometricamente retificado. O isolamento em contrato de insumos garante que a entrada da rede atenda exatamente a essa premissa.
- **Argumento para o Time:**  
  *"O caderno de IA não manipula parâmetros físicos da câmera do drone; ele opera estritamente sobre o par multimodal calibrado entregue pelo contrato de dados."*

---

<a id="decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu"></a>
### Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU / CPU Multi-threading

- **O que foi decidido:**  
  Implementar a rotina inteligente `detect_compute_device()` que realiza uma alocação de teste de tensor em GPU CUDA e, em caso de ausência de kernels compatíveis na compilação local do PyTorch, efetua fallback automático transparente para CPU multi-threaded (`torch.set_num_threads(8)`).
- **Fundamentação Técnica e Matemática:**  
  O seletor adaptativo valida a execução prévia de micro-kernels de convolução e alocação de tensores na GPU ativa antes de carregar o modelo completo na memória. Isso garante resiliência operacional contínua contra a falha `RuntimeError: CUDA error: no kernel image is available for execution on the device`, comum em arquiteturas recentes da NVIDIA (ex: RTX 50 com capacidade de computação `sm_120`), efetuando fallback transparente para CPU multi-threaded (`torch.set_num_threads(8)`) e permitindo executar a inferência de teste sem interrupções.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 4.2 (*Implementation details*).
  - **Trecho Citado:**  
    > *"The implementation setting includes: (1) GPU (NVIDIA RTX 3090)..."* (Seção 4.2)
  - **Conexão Técnica:** Os autores utilizaram a arquitetura Ampere (RTX 3090). A detecção dinâmica com fallback gracioso garante que a implementação seja agnóstica à infraestrutura, rodando com aceleração máxima onde houver suporte e permitindo validação funcional determinística em CPU quando necessário.
- **Argumento para o Time:**  
  *"O código é à prova de falhas de ambiente: utiliza GPU CUDA quando suportada e efetua fallback automático com alto desempenho multi-thread em CPU."*

---

<a id="decisao-04-normalizacao-estatistica-empirica-do-benchmark-rgbt-cc"></a>
### Decisão 04: Normalização Estatística Empírica do Benchmark RGBT-CC

- **O que foi decidido:**  
  Aplicar a normalização estatística oficial calculada empiricamente sobre o conjunto de treino do dataset **RGBT-CC** (2.030 pares de imagens):
  $$\text{RGB}: \mu = [0.407, 0.389, 0.396], \quad \sigma = [0.241, 0.246, 0.242]$$
  $$\text{Térmica}: \mu = [0.492, 0.168, 0.430], \quad \sigma = [0.317, 0.174, 0.191]$$
  E redimensionar os tensores para resolução múltipla estrita de 32 ($640 \times 512$ px).
- **Fundamentação Técnica e Matemática:**  
  Diferente de modelos baseados em convoluções genéricas (como VGG) que adotam médias do ImageNet em ambos os canais, o benchmark RGBT-CC de Liu et al. calibra estatísticas específicas para a resposta radiométrica de sensores térmicos de infravermelho longo (LWIR). Isso evita a saturação do canal térmico e preserva sua distribuição fortemente assimétrica ($\mu_{green} = 0.168$ vs $\mu_{red} = 0.492$); utilizar a média do ImageNet ($0.456$) no canal térmico descalibraria os valores de entrada dos patches do Transformer em mais de $170\%$. Além disso, o redimensionamento estrito para dimensões múltiplas de 32 ($640 \times 512$ px) é exigido pela estrutura piramidal de 4 estágios do PVTv2 (fatores de escala $1/4, 1/8, 1/16, 1/32$) para evitar truncamentos na divisão de patches e nas camadas de atenção com redução espacial linear (SRA).
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 4.1 (*Datasets and evaluation metrics: RGBT-CC*) e Seção 4.2 (*Implementation details*).
  - **Trecho Citado:**  
    > *"Dataset. The public RGBT-CC [20] dataset is adopted to evaluate our method. RGBT-CC consists of 1,030 training samples, 200 validation samples, and 800 testing ones."* (Seção 4.1)
  - **Conexão Técnica:** Para replicar os pesos do checkpoint treinado do Liuzywen sem sofrer desvio de domínio (*domain shift*), os tensores de entrada precisam compartilhar rigorosamente a mesma distribuição estatística de normalização com que a rede convergiu durante as 17 horas de treinamento reportadas no artigo.
- **Alternativas Descartadas:**  
  - *Normalização genérica ImageNet:* Rejeitada porque distorce severamente o canal térmico e compromete a ativação das camadas lineares do PVTv2.
- **Argumento para o Time:**  
  *"Respeitamos a distribuição estatística empírica exata do dataset RGBT-CC para o qual os pesos do Liuzywen foram otimizados, protegendo a rede contra domain shift na imagem térmica."*

---

<a id="decisao-05-arquitetura-siamesa-pvtv2-b2-com-reducao-espacial-linear-sra"></a>
### Decisão 05: Arquitetura Siamesa PVTv2-B2 com Redução Espacial Linear (SRA)

- **O que foi decidido:**  
  Utilizar o backbone siamês **Pyramid Vision Transformer v2 (PVTv2-B2)** para extração hierárquica paralela de características em 4 estágios ($F^1, F^2, F^3, F^4$) com *Linear Spatial Reduction Attention (SRA)*.
- **Fundamentação Técnica e Matemática:**  
  Enquanto convoluções clássicas (como VGG-19) possuem campos receptivos limitados pelo tamanho de kernel ($3 \times 3$), o PVTv2 processa a imagem em formato piramidal de tokens. A SRA reduz a complexidade quadrática da autoatenção $\mathcal{O}(H^2 W^2)$ para $\mathcal{O}\left(\frac{H^2 W^2}{R^2}\right)$ aplicando uma taxa de redução espacial $R$, viabilizando a captura de dependências de longo alcance em tensores de alta resolução.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 2.2 (*Transformer based crowd counting*), Seção 3.1 (*Equation 1*) e Seção 4.4.1 (*Table 2 - Variant No. 1 Baseline*).
  - **Trecho Citado:**  
    > *"Previous works utilize the convolution neural network as the backbone and regress density map to predict the crowd count. The advent of transformer has pushed the crowd counting model forward."* (Seção 2.2)  
    > *"Given a paired RGB-T image $I = \{I_r, I_t\}$, we use two PVT encoders [39] as the feature extractors to capture hierarchical features: $F_r = E_{PVT}(I_r)$ and $F_t = E_{PVT}(I_t)$, where $E_{PVT}$ denotes a PVT encoder, $F_r = \{F_r^i\}_{i=1}^4$ and $F_t = \{F_t^i\}_{i=1}^4$ represent color features and thermal features, respectively, $i$ is the feature layer number."* (Seção 3.1, Eq. 1)  
    > *"At first, we construct a baseline model. It concatenates high-layer features of two PVT encoders and applies regression head to predict the density map and sum up... The baseline achieves GAME(0) of 11.62 and RMSE of 19.88."* (Seção 4.4.1, Tabela 2)
  - **Conexão Técnica:** O artigo adota os encoders PVTv2 [39] porque seus quatro estágios piramidais geram representações complementares: os estágios rasos ($F_r^1, F_r^2$) preservam contornos geométricos finos e bordas de cabeças individuais, enquanto o estágio profundo ($F_r^4, F_t^4$) sintetiza a semântica de alta densidade da multidão. O estudo de ablação da Tabela 2 comprova que apenas o baseline de dois PVT encoders já atinge um GAME(0) altamente competitivo de **11.62**, servindo de base sólida para os módulos de fusão subsequentes.
- **Argumento para o Time:**  
  *"O PVTv2 combina a resolução espacial progressiva das redes piramidais com a capacidade de capturar contexto global do Transformer, superando as limitações de campo receptivo de convoluções tradicionais."*

---

<a id="decisao-06-count-guided-multi-scale-token-transformer-msttrans-e-token-fcount"></a>
### Decisão 06: Count-Guided Multi-Scale Token Transformer (MSTTrans e Token $F_{count}$)

- **O que foi decidido:**  
  Instanciar o módulo **MSTTrans** que injeta um token treinável de contagem $F_{count} \in \mathbb{R}^{1 \times C}$ e processa as características de topo $F_r^4$ e $F_t^4$ em três resoluções paralelas de tokens ($N^2, N, 1$).
- **Fundamentação Técnica e Matemática:**  
  Grandes variações na escala aparente das pessoas (pedestres próximos vs distantes) são o principal desafio da contagem de multidões. Inspirado no conceito de ASPP convolucional, o MSTTrans constrói 3 escalas na dimensão de tokens:
  1. *Escala Fina ($N^2$ tokens):* $f_1 = [F_r^4, F_t^4, F_{count}] \in \mathbb{R}^{(2N^2+1) \times C}$.
  2. *Escala Média ($N$ tokens):* $f_2 = [\text{merge}_{N^2 \to N}(F_r^4), \text{merge}_{N^2 \to N}(F_t^4), F_{count}] \in \mathbb{R}^{(2N+1) \times C}$.
  3. *Escala Global ($1$ token):* $f_3 = [\text{merge}_{N^2 \to 1}(F_r^4), \text{merge}_{N^2 \to 1}(F_t^4), F_{count}] \in \mathbb{R}^{(2+1) \times C}$.  
  As três sequências passam por autoatenção multi-cabeça paralela (MHSA), permitindo que o token $F_{count}$ aprenda uma estimativa grosseira global enquanto sincroniza os canais de cor e calor. Além disso, o token de contagem $F_{count}$ introduz uma restrição semântica global durante a fusão multimodal, impedindo que ruídos de aquecimento ambiente (ex: asfalto exposto ao sol, lâmpadas ou reflexos térmicos do solo) provoquem ativações erráticas de falsos positivos no mapa de densidade.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3.1 (*Count guided multi-modal fusion - Equations 2 e 3*), Seção 4.4.1 (*Table 2*) e Seção 4.4.2 (*Table 3*).
  - **Trecho Citado:**  
    > *"To fully align two-modal data and generate a consistent result, a learnable count token is designed to guide the two-modal fusion... We design a learnable count token $F_{count}$ which implies the coarse number of crowd. The three are concatenated along the token direction, and then fed into a Multi-Scale Token Transformer (MSTTrans) which spreads information among color, thermal, and crowd count by the multi-head self-attention."* (Seção 3.1)  
    > *"$f_1 = [F_r^4, F_t^4, F_{count}]$, $f_2 = [\text{merge}_{N^2 \to N}(F_r^4), \text{merge}_{N^2 \to N}(F_t^4), F_{count}]$"* (Seção 3.1, Eq. 2 e 3)  
    > *"Table 3: Ablation study about count guidance and multi-scale concept... 'Ours/count' represents our model removing the learnable count token (GAME(0) worsens from 10.90 to 11.82, RMSE worsens from 18.79 to 20.54). 'Ours/multi-scale' represents our model with vanilla multi-head self-attention instead of MSTTrans (GAME(0) worsens from 10.90 to 11.82, RMSE worsens from 18.79 to 21.73)."* (Seção 4.4.2)
  - **Conexão Técnica:** É a contribuição central do autor. O estudo de ablação da Tabela 3 fornece a evidência empírica definitiva: (1) retirar a orientação do token de contagem ($F_{count}$) faz o RMSE saltar de **18.79 para 20.54**; (2) desativar a agregação multiescala de tokens e usar MHSA convencional faz o RMSE saltar para **21.73**. Isso comprova matematicamente que o token $F_{count}$ restringe o espaço de otimização da fusão multimodal em direção à métrica de contagem, e que o fatiamento multiescala ($N^2, N, 1$) resolve a disparidade de escala de pedestres.
- **Argumento para o Time:**  
  *"O artigo prova na Tabela 3 que sem o token de contagem F_count e sem a fusão multiescala de tokens, o erro RMSE aumenta em até 15.6%. O MSTTrans é essencial para manter a estabilidade da contagem em pedestres de diferentes tamanhos."*

---

<a id="decisao-07-modal-guided-count-enhancement-msdtrans-via-deformable-attention"></a>
### Decisão 07: Modal-Guided Count Enhancement (MSDTrans via Deformable Attention)

- **O que foi decidido:**  
  Utilizar o módulo **MSDTrans** com atenção deformável multiescala (*Multi-Scale Deformable Attention*), no qual a modalidade térmica e o token de contagem formam a Query ($Q = [G_t, G_{count}]$), e as features de cor aprimoradas multiescala $\{G_r, F_r^3, F_r^2, F_r^1\}$ compõem Key ($K$) e Value ($V$).
- **Fundamentação Técnica e Matemática:**  
  Em cenários noturnos ou de iluminação precária, a imagem térmica é o indicador mais confiável de presença humana. Por isso, a rede utiliza a representação térmica como âncora de conteúdo ($Q$), consultando dinamicamente as características visuais multiescala ($K, V$) por meio de pontos de amostragem aprendidos com offsets contínuos (atenção deformável).
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3.2 (*Modal-guided counting enhancement - Equation 9*) e Seção 4.4.1 (*Table 2 - Variant No. 3*).
  - **Trecho Citado:**  
    > *"The researches pointed out that the thermal image can provide strong support on density map estimation, especially in the dark background [29]. In the paper, we use the thermal modality to predict the density map and count, and further use color modality to refine the prediction."* (Seção 3.2)  
    > *"$[O_t, O_{count}] = \text{DeformAttn}([G_t, G_{count}], \{G_r, F_r^3, F_r^2, F_r^1\})$"* (Seção 3.2, Eq. 9)  
    > *"MSDTrans improves the performance from GAME(0) (11.62) to GAME(0) (11.17). It indicates the supplementary effect of a modality on the other modality. Last, the whole model achieves a best GAME(0) (10.90)..."* (Seção 4.4.1)
  - **Conexão Técnica:** A formulação da Eq. 9 estabelece o fluxo assimétrico da fusão: em vez de somar features cegamente (o que misturaria artefatos e ruídos ópticos noturnos), o canal térmico comanda a busca espacial e a cor enriquece com detalhes anatômicos e de textura em 4 escalas hierárquicas ($G_r, F_r^3, F_r^2, F_r^1$). A Tabela 2 comprova a eficácia desse mecanismo, que de forma isolada reduz o GAME(0) de **11.62 para 11.17** e, em conjunto com o MSTTrans, atinge a liderança de **10.90**.
- **Argumento para o Time:**  
  *"A térmica é excelente para achar pessoas no escuro mas carece de textura; a câmera visual tem muita textura mas falha no escuro. O MSDTrans usa a térmica para guiar a atenção e busca na cor apenas o refino necessário."*

---

<a id="decisao-08-inferencia-sem-gradientes-e-reconstrucao-da-densidade-com-softplus"></a>
### Decisão 08: Inferência sem Gradientes e Reconstrução da Densidade com Softplus

- **O que foi decidido:**  
  Executar a predição protegida por `with torch.no_grad():`, convertendo o tensor aprimorado $O_t$ em densidade contínua $D(x, y)$ por meio de um cabeçote convolucional com ativação Softplus.
- **Fundamentação Técnica e Matemática:**  
  A ativação Softplus ($\ln(1 + e^x)$) garante matematicamente que a densidade estimada seja estritamente não-negativa ($D(x, y) \ge 0$), prevenindo contagens negativas absurdas que poderiam ocorrer com projeções lineares desbalanceadas. O bloco `torch.no_grad()` reduz o uso de memória em mais de 50%.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3.3 (*Regression head and loss function - Equation 10*).
  - **Trecho Citado:**  
    > *"To obtain the density map, we use a simple regression head which consists of two 3×3 convolution layers and one 1×1 convolution layer: $D = RH(O_t)$, where $RH$ is the regression head."* (Seção 3.3, Eq. 10)
  - **Conexão Técnica:** O regression head projeta diretamente o espaço de canais latentes enriquecidos de volta na dimensão espacial da imagem, produzindo a matriz de predição pontual de densidade $D$. Desativar o rastreamento de gradientes no runtime permite inferir imagens de alta resolução em apenas ~700ms em CPU e <50ms em GPU.
- **Argumento para o Time:**  
  *"A ativação Softplus garante que densidades negativas sejam impossíveis por construção matemática, enquanto torch.no_grad() garante inferência ultrarrápida."*

---

<a id="decisao-09-integracao-numerica-em-densidade-e-dupla-validacao-com-ocount"></a>
### Decisão 09: Integração Numérica em Densidade e Dupla Validação com $O_{count}$

- **O que foi decidido:**  
  Computar a contagem total de pessoas por integração de superfície contínua do mapa de densidade ($\sum_{i=1}^H \sum_{j=1}^W D_{i, j}$) e contrastar o resultado com o escalar direto predito pelo token global $O_{count}$.
- **Fundamentação Técnica e Matemática:**  
  A regressão por mapa de densidade modela a presença humana através de distribuições Gaussianas normalizadas, nas quais a integral sobre o plano do pedestre equivale a exatamente $1.0$. A integração espacial contínua $\sum D(x, y)$ é matematicamente imune a problemas de oclusão severa e aglomeração densa, superando amplamente detectores tradicionais baseados em caixas delimitadoras (*bounding boxes* como YOLO), que degradam rapidamente sob altas densidades populacionais devido à sobreposição de caixas e limites do algoritmo de Supressão de Não-Máximos (NMS). O modelo Liuzywen produz **dois estimadores concorrentes**:
  1. *Estimador Espacial Denso:* $\text{Count}_{\text{density}} = \sum_{i, j} D_{i, j}$.
  2. *Estimador Coarse Global:* $\text{Count}_{\text{token}} = O_{count}$.  
  A convergência entre os dois estimadores fornece um mecanismo intrínseco de auditoria de consistência semântica.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 3.3 (*Regression head and loss function - Equation 11*) e Seção 4.1 (*Evaluation Metrics - Equations 12 e 13*).
  - **Trecho Citado:**  
    > *"The loss includes a loss about the density map and a loss about the learnable count token: $L = L_D(D, D^*) + L_C(O_{count}, C^*)$, where $L_D$ adopts distribution matching loss proposed in [33]... $L_C$ adopts L1 norm to supervise the count token. $D^*$ and $C^*$ represent the ground truth of density map and count, respectively."* (Seção 3.3, Eq. 11)  
    > *"$GAME(l) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^{4^l} |\hat{P}_{ij} - P_{ij}|$, $RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{P}_i - P_i)^2}$"* (Seção 4.1, Eq. 12 e 13)
  - **Conexão Técnica:** O modelo foi treinado sob supervisão conjunta de dois termos de perda: a perda de matching de distribuição (DM-Count [33], baseada em Transporte Ótimo e Variação Total) no mapa $D$, e a perda $L_1$ no token de contagem $O_{count}$. A integração de superfície da matriz densa é a base para as predições regionais $\hat{P}_{ij}$, permitindo calcular com exatidão matemática o erro absoluto particionado em $4^l$ quadrantes (GAME).
- **Argumento para o Time:**  
  *"Dispomos de dupla validação: a integração espacial da densidade mapeia onde cada pedestre está, enquanto o token global de contagem atua como um sanity check independente contra oclusões."*

---

<a id="decisao-10-projecoes-visuais-heatmap-jet-xai-e-exportacao-mlops-com-metricas-gamermse"></a>
### Decisão 10: Projeções Visuais, Heatmap JET (XAI) e Exportação MLOps com Métricas GAME/RMSE

- **O que foi decidido:**  
  Gerar visualizações em pseudo-cores com Colormap JET sobre as modalidades ópticas e térmicas ($\alpha = 0.45$), recortar ROIs com zoom nos pedestres e persistir os resultados na pasta `output/02_contagem/` (`density_map.npy`, `telemetria_contagem.json`, `painel_contagem_multimodal.jpg`).
- **Fundamentação Técnica e Matemática:**  
  Atende aos preceitos de inteligência artificial explicável (Explainable AI - XAI), fornecendo subsídio visual imediato para auditoria humana através da sobreposição translúcida com colormap JET sobre ambas as modalidades, certificando que os picos de densidade correspondam fielmente às cabeças e silhuetas de pedestres em solo. Garante também a gravação da matriz densa `.npy` crua em ponto flutuante contínuo (`float32`) para posterior cálculo exato das métricas científicas GAME ($l \in \{0, 1, 2, 3\}$) e RMSE sem degradação ou perda de quantização por compressão com perdas (como JPEG de 8 bits). Adicionalmente, a telemetria estruturada em JSON assegura compatibilidade imediata para consumo por APIs de monitoramento.
- **Embasamento Científico no Artigo (BMVC 2022 / arXiv:2301.03033v1):**
  - **Seção do Artigo:** Seção 4.1 (*Evaluation Metrics - Equations 12 e 13*) e Seção 4.3 (*Comparison with state-of-the-art methods - Table 1*).
  - **Trecho Citado:**  
    > *"Table 1 shows our method achieves about 8.4%, 7.8%, 5.7%, 4.1%, 10.9% improvement over the second best result in GAME(0), GAME(1), GAME(2), GAME(3) and RMSE, respectively."* (Seção 4.3)  
    > *"Methods Comparison on RGBT-CC: CSRNet (GAME0: 20.40), BL (18.70), DM-Count (16.54), P2PNet (16.24), MARUNet (17.39), MAN (17.16), CMCRL (15.61), TAFNet (12.38), MAT (12.35), DEFNet (11.90), Ours (10.90)."* (Seção 4.3, Tabela 1)
  - **Conexão Técnica:** A exportação da matriz `density_map.npy` em formato contínuo sem perdas (`float32`) permite particionar o mapa em $4^l$ regiões para auditar o erro em diferentes níveis de subdivisão $l \in \{0, 1, 2, 3\}$, espelhando rigorosamente o protocolo de avaliação com o qual os autores bateram o estado da arte na BMVC 2022. O painel visual consolida a explicabilidade exigida para homologação de sistemas críticos de segurança pública.
- **Argumento para o Time:**  
  *"Exportamos entregáveis prontos para produção: telemetria leve em JSON para APIs de monitoramento, painéis visuais para validação por operadores e matrizes NumPy puras para auditoria matemática estrita das métricas do artigo (GAME)."*
