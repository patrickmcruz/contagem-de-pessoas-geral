# Dossiê de Defesa Técnica: Arquitetura DEF-rgbtcc, Inferência Multimodal e Contagem de Densidade
**Notebook:** [`notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb)  
**Artigo de Referência:** *A Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion* (Feng et al., arXiv 2509.17079)  
**Público-Alvo:** Engenheiros de Machine Learning, Pares Técnicos e Liderança de Projeto  
**Objetivo:** Fornecer a argumentação científica, matemática e arquitetural para defender perante o time as decisões tomadas em cada célula do modelo **DEF-rgbtcc**.

---

## Índice das Decisões
1. [Decisão 01: Setup do Ambiente e Estrutura de Imports](#decisao-01-setup-do-ambiente-e-estrutura-de-imports)
2. [Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado](#decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado)
3. [Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU](#decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu)
4. [Decisão 04: Arquitetura Neural Siamesa Dual-Stream (DEF-rgbtcc)](#decisao-04-arquitetura-neural-siamesa-dual-stream-thermalrgbnet)
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
  Importar bibliotecas de tensores e redes profundas (`torch`, `torchvision`), configurar resolução de diretórios via `pathlib.Path` e injetar a pasta `notebooks/DEF-rgbtcc` no `sys.path`.
- **Fundamentação Técnica e Matemática:**  
  Garante que os módulos de rede (`models.def_rgbtcc_net`) e as camadas de atenção espacial modulada (**SMA**) sejam importados diretamente da árvore de código do repositório sem a necessidade de instalações externas.
- **Argumento para o Time:**  
  *"Qualquer desenvolvedor que clonar o repositório conseguirá rodar o notebook imediatamente sem comandos de instalação adicionais."*

---

<a id="decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado"></a>
### Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado

- **O que foi decidido:**  
  Consumir exclusivamente os arquivos gerados em `output/01_pre_transformacao/` (`rgb_preprocessed.jpg`, `thermal_preprocessed.jpg`), rejeitando qualquer reprocessamento das fotos brutas dentro deste caderno.
- **Fundamentação Técnica e Matemática:**  
  Aplicação do princípio arquitetural de **Separação de Preocupações (Separation of Concerns - SoC)**.  
  Se o Notebook 02 tentasse recalcular cortes ou desdistorções, qualquer melhoria óptica exigiria alterar dois cadernos simultaneamente.
- **Argumento para o Time:**  
  *"O Notebook 02 não se preocupa com a física do drone; seu foco é 100% inteligência artificial."*

---

<a id="decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu"></a>
### Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU

- **O que foi decidido:**  
  Implementar a função inteligente `detect_compute_device()` que testa ativamente a execução de tensores em GPU e realiza fallback automático para CPU caso a GPU não possua kernels compilados no PyTorch local.
- **Fundamentação Técnica e Matemática:**  
  Evita a exceção fatal `RuntimeError: CUDA error: no kernel image is available for execution on the device` em placas de geração recente (como NVIDIA RTX 50 Blackwell `sm_120`).
- **Argumento para o Time:**  
  *"O código é blindado: roda com aceleração máxima em placas suportadas e faz fallback transparente para CPU em hardwares ultra-recentes."*

---

<a id="decisao-04-arquitetura-neural-siamesa-dual-stream-thermalrgbnet"></a>
### Decisão 04: Arquitetura Neural Siamesa Dual-Stream (DEF-rgbtcc)

- **O que foi decidido:**  
  Utilizar a arquitetura **DEF-rgbtcc** (arXiv 2509.17079) composta por:
  1. *Backbone Compartilhado VGG-19:* Extração paralela de características com compartilhamento de pesos (Weight Sharing), reduzindo o espaço de parâmetros e alinhando o espaço latente.
  2. *Spatially Modulated Attention (SMA):* Injeção de viés indutivo espacial 2D através de uma máscara de decaimento Euclidiano treinável $M_{ij} = (\beta'_{scale})^{S'_{ij}}$ para suprimir ruído de fundo.
  3. *Adaptive Fusion Modulation (AFM):* Mecanismo de gating dinâmico $w \in [0, 1]$ em nível de cena para priorizar a modalidade mais confiável em baixa iluminação.
- **Fundamentação Técnica e Matemática:**  
  A SMA penaliza interações distantes irrelevantes, fazendo com que cabeças de atenção especializadas capturem pedestres locais e outras capturem o contexto global da multidão. A AFM pondera $F_{fused} = w \cdot F'_r + (1-w) \cdot F'_t$, fornecendo resiliência noturna.
- **Argumento para o Time:**  
  *"A arquitetura DEF-rgbtcc resolve a dispersão de atenção em fundos complexos e ajusta dinamicamente o peso da imagem térmica quando a câmera comum fica no escuro."*

---

<a id="decisao-05-tiling-espacial-3x2-e-normalizacao-estatistica-imagenet"></a>
### Decisão 05: Tiling Espacial 3x2 e Normalização Estatística ImageNet

- **O que foi decidido:**  
  Normalizar as imagens com as médias e desvios padrão oficiais do ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Fundamentação Técnica e Matemática:**  
  Como o backbone do DEF-rgbtcc é uma VGG-19 pré-treinada na ImageNet, a normalização alinha a distribuição dos canais de entrada com os pesos originais do extrator.
- **Argumento para o Time:**  
  *"Respeitamos a distribuição estatística exata para a qual os filtros convolucionais da VGG-19 foram otimizados."*

---

<a id="decisao-06-inferencia-sem-gradientes-e-reconstrucao-do-mapa-de-densidade"></a>
### Decisão 06: Inferência sem Gradientes e Reconstrução do Mapa de Densidade

- **O que foi decidido:**  
  Executar a inferência encapsulada em bloco `with torch.no_grad():` e gerar a matriz contínua de densidade com Softplus.
- **Fundamentação Técnica e Matemática:**  
  `torch.no_grad()` desativa o grafo de autodiferenciação computacional, reduzindo o consumo de VRAM em mais de $50\%$ e acelerando a resposta.
- **Argumento para o Time:**  
  *"Inferência em produção não treina pesos; desativar gradientes deixa a resposta instantânea."*

---

<a id="decisao-07-integracao-numerica-como-estimador-de-contagem-de-multidoes"></a>
### Decisão 07: Integração Numérica como Estimador de Contagem de Multidões

- **O que foi decidido:**  
  Calcular o número final de pessoas integrando numericamente a matriz de densidade:
  $$\text{Contagem} = \iint_{\Omega} D(x, y) \, dx \, dy \approx \sum_{i=1}^{H} \sum_{j=1}^{W} D_{i, j}$$
- **Fundamentação Técnica e Matemática:**  
  Modelos de detecção por bounding box (YOLO) falham em oclusão severa. A regressão por mapa de densidade gaussiano soma a probabilidade de presença contínua, sendo imune a sobreposições.
- **Argumento para o Time:**  
  *"A integral de densidade soma a massa de presença humana, sendo matematicamente imune a pedestres sobrepostos."*

---

<a id="decisao-08-projecoes-visuais-e-colormap-jet-explainable-ai---xai"></a>
### Decisão 08: Projeções Visuais e Colormap JET (Explainable AI - XAI)

- **O que foi decidido:**  
  Gerar projeções sobrepostas utilizando `cv2.COLORMAP_JET` com transparência $\alpha = 0.45$.
- **Fundamentação Técnica e Matemática:**  
  A inteligência artificial não pode ser uma 'caixa preta'. A projeção do mapa de calor permite auditar visualmente a localização da multidão.
- **Argumento para o Time:**  
  *"Entregamos a prova visual de onde cada pessoa foi detectada, gerando transparência e confiança."*

---

<a id="decisao-09-auditoria-de-deteccao-em-condicoes-adversas-de-iluminacao"></a>
### Decisão 09: Auditoria de Detecção em Condições Adversas de Iluminação

- **O que foi decidido:**  
  Recortar a ROI de pedestres no plano do solo e comparar o canal óptico escuro com o canal térmico e a fusão AFM.
- **Fundamentação Técnica e Matemática:**  
  Comprova o valor do módulo AFM em direcionar o peso de fusão para o canal infravermelho em baixa iluminação.
- **Argumento para o Time:**  
  *"Mostramos pedestres invisíveis no escuro para a câmera comum que foram detectados pela assinatura de calor."*

---

<a id="decisao-10-exportacao-dos-entregaveis-em-pasta-dedicada-e-telemetria-json"></a>
### Decisão 10: Exportação dos Entregáveis em Pasta Dedicada e Telemetria JSON

- **O que foi decidido:**  
  Gravar os artefatos em `notebooks/DEF-rgbtcc/output/02_contagem/` incluindo `density_map.npy` e `telemetria_contagem.json`.
- **Fundamentação Técnica e Matemática:**  
  Adere às melhores práticas de MLOps: auditabilidade, reprodutibilidade e fácil integração com APIs.
- **Argumento para o Time:**  
  *"Salvamos os dados em NumPy para análise matemática, imagens JPG para relatórios e JSON para APIs."*
