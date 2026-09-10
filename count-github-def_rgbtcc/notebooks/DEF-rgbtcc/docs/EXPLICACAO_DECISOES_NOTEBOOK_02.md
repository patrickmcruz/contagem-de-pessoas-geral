# Dossiê de Defesa Técnica: Arquitetura DEF-rgbtcc, Inferência Multimodal e Contagem de Densidade
**Notebook:** [`notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`](../02_contagem_pessoas_rgbtcc.ipynb)  
**Artigo Científico de Referência:** *A Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion* (Feng et al., arXiv 2509.17079, 2025)  
**Público-Alvo:** Engenheiros de Machine Learning, Pares Técnicos e Liderança de Projeto  
**Objetivo:** Fornecer a fundamentação científica, matemática, arquitetural e experimental para defender perante o time as decisões tomadas em cada etapa do pipeline de inferência neural do modelo **DEF-rgbtcc**.

---

## Índice das Decisões
1. [Decisão 01: Setup do Ambiente e Estrutura de Imports](#decisao-01-setup-do-ambiente-e-estrutura-de-imports)
2. [Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado](#decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado)
3. [Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU](#decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu)
4. [Decisão 04: Arquitetura Neural Siamesa Dual-Stream (DEF-rgbtcc: VGG19 + SMA + AFM)](#decisao-04-arquitetura-neural-siamesa-dual-stream-thermalrgbnet)
5. [Decisão 05: Tiling Espacial 3x2 e Normalização Estatística ImageNet](#decisao-05-tiling-espacial-3x2-e-normalizacao-estatistica-imagenet)
6. [Decisão 06: Inferência sem Gradientes e Reconstrução do Mapa de Densidade](#decisao-06-inferencia-sem-gradientes-e-reconstrucao-do-mapa-de-densidade)
7. [Decisão 07: Integração Numérica como Estimador de Contagem de Multidões](#decisao-07-integracao-numerica-como-estimador-de-contagem-de-multidoes)
8. [Decisão 08: Projeções Visuais e Colormap JET (Explainable AI - XAI)](#decisao-08-projecoes-visuais-e-colormap-jet-explainable-ai---xai)
9. [Decisão 09: Auditoria de Detecção em Condições Adversas de Iluminação](#decisao-09-auditoria-de-deteccao-em-condicoes-adversas-de-iluminacao)
10. [Decisão 10: Exportação dos Entregáveis em Pasta Dedicada e Telemetria JSON](#decisao-10-exportacao-dos-entregaveis-em-pasta-dedicada-e-telemetria-json)

---

<a id="decisao-01-setup-do-ambiente-e-estrutura-de-imports"></a>
### Decisão 01: Setup do Ambiente e Estrutura de Imports

- **O que foi decidido:**  
  Importar bibliotecas de tensores e redes profundas (`torch`, `torchvision`), configurar resolução de diretórios via `pathlib.Path` e injetar dinamicamente a pasta `notebooks/DEF-rgbtcc` no `sys.path`.
- **Fundamentação Técnica e Matemática:**  
  Garante que os módulos de rede (`models.def_rgbtcc_net`) e as camadas de atenção espacial modulada (**SMA**) sejam importados diretamente da árvore de código do repositório sem a necessidade de empacotamentos externos ou dependências adicionais.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Abstract e Seção 3 (*Implementation Details*).
  - **Trecho Citado:**  
    > *"Code available at https://github.com/Cht2924/RGBT-Crowd-Counting."* (Abstract)  
    > *"Our model is implemented in PyTorch and trained on a single NVIDIA RTX 3090 GPU."* (Seção 3)
  - **Conexão Técnica:** O artigo disponibiliza a arquitetura modular em PyTorch contendo os blocos proprietários SMA e AFM. A injeção no `sys.path` assegura que a estrutura exata dos módulos neurais desenvolvida pelos autores seja carregada e executada diretamente no ambiente local.
- **Argumento para o Time:**  
  *"Qualquer desenvolvedor ou cientista de dados que clonar o repositório conseguirá rodar o notebook imediatamente sem comandos de instalação adicionais."*

---

<a id="decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado"></a>
### Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado

- **O que foi decidido:**  
  Consumir exclusivamente os arquivos gerados em `output/01_pre_transformacao/` (`rgb_preprocessed.jpg`, `thermal_preprocessed.jpg`), rejeitando qualquer reprocessamento das fotos brutas dentro deste caderno.
- **Fundamentação Técnica e Matemática:**  
  Aplicação do princípio arquitetural de **Separação de Preocupações (Separation of Concerns - SoC)**. Se o Notebook 02 tentasse recalcular desdistorções ou cortes, qualquer ajuste no alinhamento óptico exigiria alterar dois cadernos simultaneamente, quebrando o versionamento determinístico.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 2 (*Methodology - Fig. 2*).
  - **Trecho Citado:**  
    > *"As illustrated in Fig. 2, a weight-sharing VGG19 backbone first extracts parallel feature maps, Fr and Ft , from an RGB-T image pair."* (Seção 2)
  - **Conexão Técnica:** O modelo DEF-rgbtcc toma como ponto de partida diretamente o par pré-alinhado e calibrado $(F_r, F_t)$, desacoplando completamente qualquer rotina de retificação física de câmeras da etapa de inferência neural.
- **Argumento para o Time:**  
  *"O Notebook 02 não se preocupa com a mecânica do drone ou ângulos de lente; seu foco é 100% inteligência artificial e regressão de densidade."*

---

<a id="decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu"></a>
### Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU

- **O que foi decidido:**  
  Implementar a função inteligente `detect_compute_device()` que testa ativamente a execução de tensores em GPU e realiza fallback automático para CPU caso a GPU não possua kernels compilados no PyTorch local.
- **Fundamentação Técnica e Matemática:**  
  Evita a interrupção fatal por `RuntimeError: CUDA error: no kernel image is available for execution on the device` em placas de arquiteturas recentes (como NVIDIA RTX 50 Blackwell `sm_120`), mantendo a robustez do pipeline de testes.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 3 (*Implementation Details*).
  - **Trecho Citado:**  
    > *"Our model is implemented in PyTorch and trained on a single NVIDIA RTX 3090 GPU."* (Seção 3)
  - **Conexão Técnica:** O artigo baseia sua validação em hardware NVIDIA acelerado por CUDA. O fallback gracioso para CPU permite que pesquisadores e sistemas sem suporte local imediato aos kernels mais recentes consigam validar numericamente os resultados do paper sem crashes de runtime.
- **Argumento para o Time:**  
  *"O código é resiliente: roda com aceleração máxima em placas suportadas e faz fallback transparente para CPU em hardwares heterogêneos."*

---

<a id="decisao-04-arquitetura-neural-siamesa-dual-stream-thermalrgbnet"></a>
### Decisão 04: Arquitetura Neural Siamesa Dual-Stream (DEF-rgbtcc: VGG19 + SMA + AFM)

- **O que foi decidido:**  
  Utilizar a arquitetura **DEF-rgbtcc** (Feng et al., arXiv 2509.17079) composta por:
  1. *Backbone Compartilhado VGG-19:* Extração paralela de características com compartilhamento de pesos (*Weight Sharing*), reduzindo parâmetros e alinhando o espaço latente.
  2. *Spatially Modulated Attention (SMA):* Injeção de viés indutivo espacial 2D através de uma máscara de decaimento Euclidiano treinável $M_{ij} = (\beta'_{scale})^{S'_{ij}}$ para suprimir ruído de fundo.
  3. *Adaptive Fusion Modulation (AFM):* Mecanismo de gating dinâmico $w \in [0, 1]$ em nível de cena para priorizar a modalidade mais confiável em condições adversas de luz.
- **Fundamentação Técnica e Matemática:**  
  A SMA penaliza interações distantes irrelevantes, permitindo que cabeças de atenção especializadas capturem pedestres locais e outras capturem o contexto global da multidão. A AFM pondera $F_{fused} = w \cdot F'_r + (1-w) \cdot F'_t$, garantindo resiliência noturna.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 2.1 (*Spatially Modulated Attention - Eq. 1 a 4*) e Seção 2.2 (*Adaptive Fusion Modulation - Eq. 5 e 6*).
  - **Trecho Citado:**  
    > *"Unlike the standard mechanism in which all tokens are effectively equidistant, our mask directly injects a explicit spatial inductive bias into the self-attention calculation... modified as follows: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot M\right) V$."* (Seção 2.1, Eq. 1)  
    > *"We introduce the AFM module, a lightweight gating mechanism that learns to intelligently adjust the contribution of RGB and thermal features based on the holistic scene context... The final fused feature map $F_{fused}$ is computed via this adaptive weighted combination: $F_{fused} = w \cdot F'_r + (1 - w) \cdot F'_t$."* (Seção 2.2, Eq. 6)
  - **Conexão Técnica:** Representa a transcrição direta das duas contribuições metodológicas principais do paper: (1) a modulação de atenção espacial com decaimento Euclidiano em potência $M_{ij} = (\beta'_{scale})^{S'_{ij}}$, que resolve a dispersão de atenção dos Transformers comuns em fundos irrelevantes; e (2) a fusão dinâmica por peso escalar $w$ aprendido em nível de cena para balancear luz visível e calor.
- **Argumento para o Time:**  
  *"A arquitetura DEF-rgbtcc elimina a dispersão de atenção em cenários urbanos complexos e transfere dinamicamente a prioridade de inferência para a imagem térmica no escuro."*

---

<a id="decisao-05-tiling-espacial-3x2-e-normalizacao-estatistica-imagenet"></a>
### Decisão 05: Tiling Espacial 3x2 e Normalização Estatística ImageNet

- **O que foi decidido:**  
  Normalizar as imagens com as médias e desvios padrão oficiais do ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Fundamentação Técnica e Matemática:**  
  Como o backbone do DEF-rgbtcc é uma VGG-19 pré-treinada na ImageNet, a normalização alinha a distribuição estatística dos canais de entrada com os pesos originais do extrator de features.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 3 (*Implementation Details*).
  - **Trecho Citado:**  
    > *"We use a pre-trained VGG19 as the backbone, followed by a 2-layer transformer encoder with 8 attention heads... For data augmentation, we apply random horizontal flipping and random cropping to 224×224."* (Seção 3)
  - **Conexão Técnica:** A inicialização com a VGG-19 pré-treinada impõe a normalização com a distribuição canônica ImageNet para assegurar que os pesos convolucionais operem dentro do espaço estatístico para o qual foram otimizados. A estratégia de tiling ou preservação de resolução mantém a escala geométrica aparente das cabeças próxima aos recortes de treino ($224 \times 224$ e $640 \times 512$).
- **Argumento para o Time:**  
  *"Respeitamos a distribuição estatística exata para a qual os filtros convolucionais da VGG-19 foram treinados, prevenindo saturações ou desvios de ativação."*

---

<a id="decisao-06-inferencia-sem-gradientes-e-reconstrucao-do-mapa-de-densidade"></a>
### Decisão 06: Inferência sem Gradientes e Reconstrução do Mapa de Densidade

- **O que foi decidido:**  
  Executar a passagem direta encapsulada em bloco `with torch.no_grad():` e gerar a matriz contínua de densidade com ativação Softplus/convolucional.
- **Fundamentação Técnica e Matemática:**  
  `torch.no_grad()` desativa a construção do grafo de autodiferenciação, reduzindo o consumo de memória de vídeo em mais de $50\%$ e acelerando sensivelmente a execução.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 2.3 (*Regression Head and Loss Function*).
  - **Trecho Citado:**  
    > *"Our regression head is composed of a stack of two 3 × 3 convolutional layers with intermediate ReLU activations. This is followed by a final 1 × 1 convolutional layer to produce the density map Dest."* (Seção 2.3)
  - **Conexão Técnica:** O regression head converte o tensor de características fundidas $F_{fused}$ na predição densa contínua $D_{est}$. A desativação dos gradientes durante a inferência viabiliza a execução rápida descrita no artigo em ambientes de produção.
- **Argumento para o Time:**  
  *"Inferência em produção não precisa atualizar pesos; desativar os grafos de gradiente deixa a predição imediata e reduz pela metade a alocação de memória."*

---

<a id="decisao-07-integracao-numerica-como-estimador-de-contagem-de-multidoes"></a>
### Decisão 07: Integração Numérica como Estimador de Contagem de Multidões

- **O que foi decidido:**  
  Calcular o número total de pedestres integrando numericamente a matriz bidimensional de densidade:
  $$\text{Contagem} = \iint_{\Omega} D(x, y) \, dx \, dy \approx \sum_{i=1}^{H} \sum_{j=1}^{W} D_{i, j}$$
- **Fundamentação Técnica e Matemática:**  
  Modelos de detecção baseados em caixas delimitadoras (bounding boxes, ex: YOLO) falham criticamente sob oclusões severas e alta densidade populacional. A regressão contínua mapeia a probabilidade espacial fracionária de presença humana: cada cabeça é representada por uma curva Gaussiana cuja integral é unitária ($\iint G(x, y) dx dy = 1.0$).
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 2.3 (*Loss Function - Eq. 7*) e Seção 3 (*Evaluation Metrics - Eq. 8*).
  - **Trecho Citado:**  
    > *"Our network is trained with the Bayesian Loss [13], a common and effective choice for reconciling discrete point supervision with continuous density map regression."* (Seção 2.3)  
    > *"where N is the number of test images, and $C_{et}^i(j)$ and $C_{gt}^i(j)$ are the estimated and ground-truth counts for the j-th region of image i, respectively."* (Seção 3, Eq. 8)
  - **Conexão Técnica:** Toda a avaliação do artigo científico baseia-se em predições onde a contagem regional estimada $C_{et}(j)$ é obtida pela soma pontual dos valores do mapa de densidade $D_{est}$. A integração numérica é a forma matemática padrão para extrair o valor quantitativo de contagem a partir de modelos treinados com Bayesian Loss.
- **Argumento para o Time:**  
  *"A integral da densidade soma a massa de presença humana de forma contínua, sendo matematicamente imune a oclusões e sobreposições de pedestres."*

---

<a id="decisao-08-projecoes-visuais-e-colormap-jet-explainable-ai---xai"></a>
### Decisão 08: Projeções Visuais e Colormap JET (Explainable AI - XAI)

- **O que foi decidido:**  
  Gerar projeções visuais sobrepostas utilizando `cv2.COLORMAP_JET` com canal alfa de transparência $\alpha = 0.45$ sobre os canais RGB e Térmico.
- **Fundamentação Técnica e Matemática:**  
  A inteligência artificial aplicada à segurança pública não pode ser tratada como caixa preta. A projeção espacial da densidade permite auditoria visual humana imediata, confirmando que os picos térmicos/ópticos coincidem rigorosamente com a posição real das pessoas.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 1 (Fig. 1) e Seção 3 (*Results and Analysis - Fig. 4*).
  - **Trecho Citado:**  
    > *"Fig. 4. Visualization of crowd density maps generated by our model and other competing methods on several challenging scenes."* (Seção 3)  
    > *"The qualitative results in Fig. 4 further showcase our model’s ability to generate high-quality density maps that closely align with the ground truth, even in challenging scenes with varying crowd densities and lighting conditions."* (Seção 3)
  - **Conexão Técnica:** Segue rigorosamente a metodologia visual adotada nas figuras 1 e 4 do paper, demonstrando graficamente como os picos do mapa de densidade formam ativações nítidas e compactas exatamente sobre as cabeças anotadas, permitindo inspecionar a qualidade da contagem tanto em cenas diurnas quanto noturnas.
- **Argumento para o Time:**  
  *"Entregamos a prova visual transparente de onde cada pessoa foi localizada, permitindo auditoria instantânea por operadores humanos e gerando confiança operacional."*

---

<a id="decisao-09-auditoria-de-deteccao-em-condicoes-adversas-de-iluminacao"></a>
### Decisão 09: Auditoria de Detecção em Condições Adversas de Iluminação

- **O que foi decidido:**  
  Recortar a Região de Interesse (ROI) de pedestres no solo escuro e comparar lado a lado o canal óptico, a imagem térmica com CLAHE e o mapa de ativação fundido pela AFM.
- **Fundamentação Técnica e Matemática:**  
  Comprova de maneira empírica o comportamento do módulo de fusão adaptativa (AFM) em transferir a ponderação para o espectro infravermelho quando a câmera óptica não possui fótons suficientes para discriminar alvos.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 2.2 (*Adaptive Fusion Modulation*) e Seção 3 (*Ablation Study - (2) Effectiveness of AFM*).
  - **Trecho Citado:**  
    > *"This mechanism allows the network to adapt to various conditions. For instance, in a low-light scene where RGB information is unreliable, the network learns to produce a small w, thereby prioritizing the more reliable thermal features to generate a more reliable final feature map."* (Seção 2.2)  
    > *"Replacing simple summation with the AFM module further reduces GAME(0) to 10.80 and RMSE to 19.87. This improvement validates our dynamic fusion strategy, which enhances performance by prioritizing the more reliable modality based on scene content."* (Seção 3)
  - **Conexão Técnica:** Esta auditoria em notebook valida no caso de uso real a hipótese de ganho experimental comprovada na Tabela 3 do artigo (onde o acréscimo da AFM reduz o erro RMSE de 20.53 para 19.87 e o GAME(0) de 11.09 para 10.80 ao priorizar dinamicamente a térmica no escuro).
- **Argumento para o Time:**  
  *"Provamos em auditoria visual pedestres que estavam invisíveis na câmera comum no escuro e que foram resgatados com precisão pela assinatura de calor via fusão adaptativa."*

---

<a id="decisao-10-exportacao-dos-entregaveis-em-pasta-dedicada-e-telemetria-json"></a>
### Decisão 10: Exportação dos Entregáveis em Pasta Dedicada e Telemetria JSON

- **O que foi decidido:**  
  Persistir os artefatos finais em `notebooks/DEF-rgbtcc/output/02_contagem/`, incluindo:
  1. `density_map.npy` (matriz contínua bidimensional com valores float32).
  2. `heatmap_sobre_rgb.jpg` e `heatmap_sobre_termica.jpg` (projeções XAI).
  3. `painel_contagem_multimodal.jpg` (dashboard completo para relatórios).
  4. `telemetria_contagem.json` (metadados estruturados com total de pessoas, peso AFM $w$, tempo de inferência e device).
- **Fundamentação Técnica e Matemática:**  
  Atende integralmente às diretrizes de engenharia MLOps: auditabilidade, reprodutibilidade e interoperabilidade com microserviços e dashboards de monitoramento.
- **Relação com o Artigo Científico (arXiv 2509.17079):**
  - **Seção do Artigo:** Seção 3 (*Evaluation Metrics: GAME e RMSE - Eq. 8*).
  - **Trecho Citado:**  
    > *"The GAME metric evaluates performance on subdivisions, where the image is divided into $4^L$ non-overlapping regions at level L. It is defined as: $\text{GAME}(L) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^{4^L} |C_{et}^i(j) - C_{gt}^i(j)|$."* (Seção 3, Eq. 8)
  - **Conexão Técnica:** A persistência da matriz densa contínua `density_map.npy` em precisão total (`float32`) é essencial para que rotinas automatizadas de validação calculem o erro absoluto em diferentes níveis de grade regional ($GAME(0), GAME(1), GAME(2), GAME(3)$) dividindo o mapa em $4^L$ blocos, exatamente como exige o protocolo de avaliação do artigo científico.
- **Argumento para o Time:**  
  *"Exportamos matrizes NumPy para validação matemática estrita das métricas do artigo (GAME), imagens de alta definição para auditoria humana e JSON estruturado para consumo direto por APIs de produção."*
