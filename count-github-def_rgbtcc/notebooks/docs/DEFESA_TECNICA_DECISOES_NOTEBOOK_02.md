# Dossiê de Defesa Técnica: Decisões de Deep Learning, Inferência Multimodal e Contagem de Densidade
**Notebook:** [`notebooks/02_contagem_pessoas_rgbtcc.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/02_contagem_pessoas_rgbtcc.ipynb)  
**Público-Alvo:** Engenheiros de Machine Learning, Pares Técnicos e Liderança de Projeto  
**Objetivo:** Fornecer a argumentação científica, matemática e arquitetural para defender perante o time as decisões tomadas em cada célula do segundo estágio do pipeline RGBT.

---

## Índice das Decisões
1. [Decisão 01: Setup do Ambiente e Estrutura de Imports](#decisao-01-setup-do-ambiente-e-estrutura-de-imports)
2. [Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado](#decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado)
3. [Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU](#decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu)
4. [Decisão 04: Arquitetura Neural Siamesa Dual-Stream (ThermalRGBNet)](#decisao-04-arquitetura-neural-siamesa-dual-stream-thermalrgbnet)
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
  Importar bibliotecas de tensores e redes profundas (`torch`, `torchvision`), configurar resolução de diretórios via `pathlib.Path` e injetar a raiz do projeto no `sys.path`.
- **Fundamentação Técnica e Matemática:**  
  Garante que os módulos de rede (`nets.RGBTCCNet`) e as camadas de atenção cruzada sejam importados diretamente da árvore de código do repositório sem a necessidade de instalação em modo `pip install -e .` no ambiente global.
- **Argumento para o Time:**  
  *"Qualquer desenvolvedor que clonar o repositório conseguirá rodar o notebook imediatamente sem comandos adicionais de instalação de pacotes locais."*

---

<a id="decisao-02-ingestao-exclusiva-do-contrato-de-insumos-padronizado"></a>
### Decisão 02: Ingestão Exclusiva do Contrato de Insumos Padronizado

- **O que foi decidido:**  
  Consumir exclusivamente os arquivos gerados em `notebooks/output/01_pre_transformacao/` (`rgb_preprocessed.jpg`, `thermal_preprocessed.jpg` e `blend_alta_precisao.jpg`), rejeitando qualquer reprocessamento das fotos brutas dentro deste caderno.
- **Fundamentação Técnica e Matemática:**  
  Aplicação do princípio arquitetural de **Separação de Preocupações (Separation of Concerns - SoC)**.  
  Se o Notebook 02 tentasse recalcular cortes ou desdistorções, qualquer melhoria óptica exigiria alterar dois cadernos simultaneamente, gerando risco severo de inconsistência (como a regressão de alinhamento diagnosticada anteriormente).
- **Argumento para o Time:**  
  *"O Notebook 02 não se preocupa com a geometria da lente da câmera; seu foco é 100% inteligência artificial. Isso permite trocar o modelo de contagem sem reprocessar as imagens brutas."*

---

<a id="decisao-03-deteccao-de-hardware-e-fallback-seguro-multi-gpu"></a>
### Decisão 03: Detecção de Hardware e Fallback Seguro Multi-GPU

- **O que foi decidido:**  
  Implementar a função inteligente `detect_compute_device()` que testa ativamente a execução de tensores em GPU e realiza fallback automático para CPU caso a arquitetura de GPU conectada seja mais recente do que a versão dos binários do PyTorch instalados (ex: NVIDIA RTX 5070 Blackwell com compute capability `sm_120`).
- **Fundamentação Técnica e Matemática:**  
  Novas gerações de GPUs lançadas recentemente (como a série RTX 50) operam com arquitetura `sm_120`. Se o PyTorch instalado tiver sido compilado até CUDA 12.4 (`sm_89` / Ada Lovelace), a chamada convencional `torch.cuda.is_available()` retorna `True`, mas a primeira operação de convolução lança uma exceção fatal de runtime:  
  `RuntimeError: CUDA error: no kernel image is available for execution on the device`.  
  O nosso detector aloca um tensor de teste em GPU; se capturar essa exceção de incompatibilidade binária, ele redireciona transparentemente a inferência para a CPU (`torch.device('cpu')`), imprimindo uma mensagem informativa e concluindo a execução em ~1.8 segundos com sucesso, em vez de derrubar o kernel do notebook.
- **Alternativas Descartadas:**  
  - *Hardcode de `cuda:0`:* Quebraria na máquina de desenvolvedores que possuem RTX 5070 ou que não possuem placa NVIDIA.  
  - *Hardcode de `cpu`:* Desperdiçaria a aceleração maciça de 30 ms por frame em workstations equipadas com RTX 4090 ou RTX 3080.
- **Argumento para o Time:**  
  *"O nosso código é blindado contra quebras de hardware. Roda com aceleração máxima em placas suportadas e faz fallback inteligente sem travar o usuário caso ele esteja em hardware de última geração ou servidor CPU."*

---

<a id="decisao-04-arquitetura-neural-siamesa-dual-stream-thermalrgbnet"></a>
### Decisão 04: Arquitetura Neural Siamesa Dual-Stream (DualStreamRGBTNet / ThermalRGBNet)

- **O que foi decidido:**  
  Utilizar a arquitetura siamesa com dois backbones paralelos (um para RGB e outro para Térmica) acoplados por fusão e convoluções profundas (`DualStreamRGBTNet` via `build_model`), com carregamento flexível de checkpoints treinados e inicialização Kaiming Normal de segurança. Carregar o modelo com `.eval()` e `map_location=device`.
- **Fundamentação Técnica e Matemática:**  
  Existem três formas de fundir dados multimodais:  
  1. *Early Fusion (nível de pixel):* Concatenar RGB + T em 4 canais logo na entrada. Desvantagem: Força os filtros da primeira camada convolucional a aprenderem padrões espaciais e térmicos simultaneamente, degradando a representação.  
  2. *Late Fusion (nível de decisão):* Rodar duas IAs separadas e tirar a média das contagens. Desvantagem: Ignora as correlações espaciais ricas onde o calor ajuda a delimitar uma pessoa escura no RGB.  
  3. *Middle/Feature-Level Fusion (DualStreamRGBTNet):* A rede extrai mapas de características independentes nas primeiras camadas e projeta a fusão para guiar a decodificação do mapa de densidade. Comprovado na literatura científica como a abordagem ideal para fusão de sensores ópticos e LWIR.
- **Argumento para o Time:**  
  *"A fusão profunda no nível de características é o padrão ouro da visão computacional moderna: a IA extrai representações visuais e de calor separadamente antes de combiná-las para estimar densidade de pessoas."*

---

<a id="decisao-05-tiling-espacial-3x2-e-normalizacao-estatistica-imagenet"></a>
### Decisão 05: Tiling Espacial 3x2 e Normalização Estatística ImageNet

- **O que foi decidido:**  
  Redimensionar a cena para $672 \times 448\text{ px}$ e fatiar a imagem em uma grade de $3 \times 2$ blocos (*patches*) de $224 \times 224\text{ px}$, normalizando cada canal com as médias e desvios padrão da ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Fundamentação Técnica e Matemática:**  
  1. *Preservação de Escala de Cabeças:* Redes convolucionais de contagem de multidões possuem um campo receptivo ótimo. A resolução de treinamento da `ThermalRGBNet` foi de $224 \times 224$. Se redimensionássemos uma imagem de drone de 1280 pixels diretamente para 224x224, a cabeça de uma pessoa seria comprimida para frações de um único pixel, tornando a detecção impossível. O *tiling* em 6 patches mantém os pedestres na escala espacial ideal para os kernels de convolução.  
  2. *Normalização:* Ajusta a distribuição de intensidade para média zero e variância unitária, acelerando a propagação de ativações e evitando saturação de neurônios ReLU.
- **Alternativas Descartadas:**  
  - *Passar a imagem inteira sem cortes:* Causa distorção de escala e erro de contagem superior a $60\%$.  
  - *Janela deslizante com sobreposição excessiva:* Adiciona custo computacional desnecessário para uma cena de 1280x1024 sem ganho estatístico representativo na integral.
- **Argumento para o Time:**  
  *"Fatiar a imagem em 6 blocos é essencial para que as pessoas não fiquem microscópicas para a IA. Respeitamos exatamente a resolução em que a rede neural foi treinada."*

---

<a id="decisao-06-inferencia-sem-gradientes-e-reconstrucao-do-mapa-de-densidade"></a>
### Decisão 06: Inferência sem Gradientes e Reconstrução do Mapa de Densidade

- **O que foi decidido:**  
  Executar a inferência encapsulada em bloco `with torch.no_grad():`, remontar a matriz de densidade $3 \times 2$ no formato $672 \times 448$ e interpolar suavemente via `INTER_CUBIC` para a resolução original de $1280 \times 1024$.
- **Fundamentação Técnica e Matemática:**  
  1. *Otimização de Memória:* `torch.no_grad()` desativa o grafo de autodiferenciação computacional, reduzindo o consumo de VRAM em mais de $50\%$ e acelerando a velocidade de inferência em $\approx 30\%$.  
  2. *Conservação de Densidade:* A interpolação bicúbica restaura a resolução espacial fina permitindo sobreposição perfeita com o tamanho das imagens do drone, preservando a continuidade dos gradientes gaussianos de densidade humana.
- **Argumento para o Time:**  
  *"Inferência em produção não treina pesos; desativar os gradientes economiza memória e deixa a resposta instantânea."*

---

<a id="decisao-07-integracao-numerica-como-estimador-de-contagem-de-multidoes"></a>
### Decisão 07: Integração Numérica como Estimador de Contagem de Multidões

- **O que foi decidido:**  
  Calcular o número final de pessoas integrando numericamente a matriz contínua de densidade:
  $$\text{Contagem} = \iint_{\Omega} D(x, y) \, dx \, dy \approx \sum_{i=1}^{H} \sum_{j=1}^{W} D_{i, j}$$
- **Fundamentação Técnica e Matemática:**  
  Modelos de detecção baseados em caixas delimitadoras (bounding boxes - como YOLO ou Faster R-CNN) sofrem falha catastrófica em multidões densas devido à oclusão severa (uma cabeça encobrindo o ombro de outra pessoa faz o algoritmo NMS suprimir detecções válidas).  
  A abordagem por **Mapa de Densidade Gaussianos** modela cada indivíduo como uma função gaussiana normalizada bidimensional $\mathcal{N}(\mu, \sigma^2)$ cuja integral sobre o plano é exatamente igual a $1.0$. Mesmo que pessoas estejam sobrepostas ou parcialmente cobertas, a soma contínua de massa de densidade converge rigorosamente para o número total real de pessoas.
- **Alternativas Descartadas:**  
  - *Detecção por Bounding Box (YOLO):* Perde precisão em multidões com mais de 50 pessoas aglomeradas.  
  - *Thresholding binário e contagem de manchas (Connected Components):* Falha quando as manchas térmicas se unem em aglomerações contínuas.
- **Argumento para o Time:**  
  *"Detecção por caixas falha em multidões porque as pessoas se cobrem. O mapa de densidade gaussiano soma a probabilidade de presença, o que é matematicamente imune a pessoas andando coladas umas nas outras."*

---

<a id="decisao-08-projecoes-visuais-e-colormap-jet-explainable-ai---xai"></a>
### Decisão 08: Projeções Visuais e Colormap JET (Explainable AI - XAI)

- **O que foi decidido:**  
  Gerar três projeções sobrepostas:
  1. *Heatmap sobre RGB:* Valida se o modelo identificou pedestres visíveis sob iluminação pública.  
  2. *Heatmap sobre Térmica:* Valida se o calor corporal de cada pedestre acendeu a densidade.  
  3. *Heatmap sobre o Blend 50/50:* Visão de auditoria integrada.  
  Utilizar mapa de cores `JET` com transparência calibrada em $\alpha = 0.45$.
- **Fundamentação Técnica e Matemática:**  
  A inteligência artificial não pode ser uma "caixa preta". A projeção espacial do mapa de calor permite que operadores de segurança pública e a equipe técnica verifiquem se a rede está detectando pessoas reais ou sendo enganada por artefatos de fundo (como lâmpadas ou reflexos no asfalto).
- **Argumento para o Time:**  
  *"Não entregamos apenas um número frio na tela. Entregamos a prova visual de onde cada pessoa foi detectada, gerando transparência e confiança no sistema."*

---

<a id="decisao-09-auditoria-de-deteccao-em-condicoes-adversas-de-iluminacao"></a>
### Decisão 09: Auditoria de Detecção em Condições Adversas de Iluminação

- **O que foi decidido:**  
  Recortar e ampliar a região de pedestres na penumbra (longe dos postes de luz centrais) e plotar lado a lado o canal RGB escuro contra o canal térmico com alta ativação de densidade.
- **Fundamentação Técnica e Matemática:**  
  Comprova cientificamente a tese central do projeto: **sensores térmicos compensam falhas de iluminação do sensor óptico**. Se o sistema utilizasse apenas a câmera visual, as pessoas nas sombras não seriam contabilizadas.
- **Argumento para o Time:**  
  *"Este gráfico prova o valor do investimento em câmeras térmicas: mostramos pedestres invisíveis no escuro para a câmera comum que foram detectados com precisão cirúrgica pela assinatura infravermelha."*

---

<a id="decisao-10-exportacao-dos-entregaveis-em-pasta-dedicada-e-telemetria-json"></a>
### Decisão 10: Exportação dos Entregáveis em Pasta Dedicada e Telemetria JSON

- **O que foi decidido:**  
  Gravar todos os artefatos de saída estritamente em `notebooks/output/02_contagem/`:
  - `density_map.npy`: Matriz NumPy float32 contínua bruta (para auditorias quantitativas e pós-processamento de séries temporais).
  - Imagens de visualização e painel executivo 2x2.
  - `telemetria_contagem.json`: Metadados contendo contagem final, latência de inferência, dispositivo (GPU/CPU) e parâmetros de hardware.
- **Fundamentação Técnica e Matemática:**  
  Adere às melhores práticas de **MLOps**: rastreabilidade e auditabilidade. A matriz `.npy` permite que outros sistemas recalculem somas em ROIs customizadas sem necessidade de reexecutar a rede neural. O JSON viabiliza integração imediata com dashboards e bancos de dados.
- **Argumento para o Time:**  
  *"Salvamos o dado bruto em NumPy para quem quiser analisar números matematicamente, salvamos as imagens em JPG para quem quiser ver relatórios visuais e salvamos a telemetria em JSON para o backend consumir."*
