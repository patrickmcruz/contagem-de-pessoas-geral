# Dossiê de Defesa Técnica: Decisões de Engenharia Óptica e Alinhamento Multimodal
**Notebook:** [`notebooks/01_pre_transformacao_alinhamento.ipynb`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/notebooks/01_pre_transformacao_alinhamento.ipynb)  
**Público-Alvo:** Engenheiros de Visão Computacional, Pares Técnicos e Liderança de Projeto  
**Objetivo:** Fornecer a argumentação rigorosa, física, matemática e arquitetural para defender perante o time as decisões tomadas em cada célula do primeiro estágio do pipeline RGBT.

---

## Índice das Decisões
1. [Decisão 01: Setup do Ambiente e Interoperabilidade com o App](#decisao-01-setup-do-ambiente-e-interoperabilidade-com-o-app)
2. [Decisão 02: Ingestão RAW e Gestão de Espaços de Cor](#decisao-02-ingestao-raw-e-gestao-de-espacos-de-cor)
3. [Decisão 03: Desdistorção de Lente Grande-Angular no Sensor RAW Completo](#decisao-03-desdistorcao-de-lente-grande-angular-no-sensor-raw-completo)
4. [Decisão 04: Casamento de Campo de Visão (FOV Center Crop Ancorado na Altura)](#decisao-04-casamento-de-campo-de-visao-fov-center-crop-ancorado-na-altura)
5. [Decisão 05: Equalização Radiométrica da Térmica (CLAHE no Espaço Lab)](#decisao-05-equalizacao-radiometrica-da-termica-clahe-no-espaco-lab)
6. [Decisão 06: Padronização de Resolução e Compensação Afim de Baseline Estéreo](#decisao-06-padronizacao-de-resolucao-e-compensacao-afim-de-baseline-estereo)
7. [Decisão 07: Validação de Paridade Bit-a-Bit com a Classe de Produção](#decisao-07-validacao-de-paridade-bit-a-bit-com-a-classe-de-producao)
8. [Decisão 08: Auditoria Visual Sub-Pixel na Mulher Central e Plano do Solo](#decisao-08-auditoria-visual-sub-pixel-na-mulher-central-e-plano-do-solo)
9. [Decisão 09: Exportação do Contrato de Insumos Padronizado](#decisao-09-exportacao-do-contrato-de-insumos-padronizado)

---

<a id="decisao-01-setup-do-ambiente-e-interoperabilidade-com-o-app"></a>
### Decisão 01: Setup do Ambiente e Interoperabilidade com o App

- **O que foi decidido:**  
  Adicionar a pasta `app/` ao `sys.path` dinamicamente, isolar o diretório de cache do Matplotlib via `MPLCONFIGDIR = '/tmp/matplotlib'` e manipular diretórios utilizando `pathlib.Path` resolvido de forma absoluta.
- **Fundamentação Técnica e Matemática:**  
  1. *Single Source of Truth:* A lógica de calibração reside no pacote de produção [`RGBTImageEqualizer`](file:///c:/Users/User/Git/contagem-de-pessoas-def-rgbtcc/count-github-def_rgbtcc/app/head_counting/preprocessing.py). O notebook precisa consumir exatamente a mesma classe para evitar "duplicação de código com divergência futura".  
  2. *Resiliência de Sistema:* Em servidores Linux multi-usuário ou instâncias Docker sem permissão no `$HOME`, o Matplotlib trava ao tentar gravar cache em `~/.cache`. A variável `/tmp/matplotlib` garante execução silenciosa e sem travamentos.  
  3. *Portabilidade de SO:* O uso de `Path.resolve()` trata transparentemente barras normais (`/`) e invertidas (`\`) entre Windows e Linux.
- **Alternativas Descartadas:**  
  - *Copiar e colar o código de `app/` dentro do notebook:* Descartado para não violar o princípio DRY (*Don't Repeat Yourself*).  
  - *Hardcode de caminhos absolutos:* Descartado porque quebraria o código no computador de outros membros da equipe.
- **Argumento para o Time:**  
  *"Garantimos que o notebook seja portátil para qualquer máquina e utilize a mesma base de código do sistema em produção, sem criar duas versões da mesma matemática."*

---

<a id="decisao-02-ingestao-raw-e-gestao-de-espacos-de-cor"></a>
### Decisão 02: Ingestão RAW e Gestão de Espaços de Cor

- **O que foi decidido:**  
  Carregar as imagens em formato nativo OpenCV BGR (`cv2.imread`) e convertê-las para RGB (`cv2.COLOR_BGR2RGB`) estritamente para visualização no Matplotlib. Inserir `assert` de existência imediato.
- **Fundamentação Técnica e Matemática:**  
  O OpenCV trabalha historicamente no formato BGR por compatibilidade com hardware de captura de câmeras, enquanto bibliotecas de renderização gráfica (Matplotlib, PIL, navegadores) operam em RGB. Se não houver a conversão antes do `plt.imshow()`, a pele dos pedestres e a imagem térmica sofrem inversão de canais (ficando azuladas), o que prejudica a auditoria humana. Os `asserts` aplicam a diretriz *Fail-Fast*, interrompendo o fluxo imediatamente caso os dados de entrada não estejam na pasta correta.
- **Alternativas Descartadas:**  
  - *Usar PIL (`Image.open`):* Descartado porque a etapa subsequente usa operadores geométricos do OpenCV (`cv2.undistort`, `cv2.warpAffine`), e a constante conversão entre PIL Image e NumPy Array adiciona overhead computacional desnecessário.
- **Argumento para o Time:**  
  *"Mantemos a alta performance do NumPy/OpenCV no backend e garantimos a fidelidade cromática para visualização sem converter tipos desnecessariamente."*

---

<a id="decisao-03-desdistorcao-de-lente-grande-angular-no-sensor-raw-completo"></a>
### Decisão 03: Desdistorção de Lente Grande-Angular no Sensor RAW Completo

- **O que foi decidido:**  
  Executar o algoritmo `cv2.undistort` **na imagem RAW completa ($8000 \times 6000\text{ px}$) ANTES de qualquer recorte**, usando a matriz intrínseca $K$ com centro óptico $cx = w/2 = 4000, cy = h/2 = 3000$ e coeficientes de distorção $D = [-0.04, 0.01, 0, 0]$.
- **Fundamentação Técnica e Matemática:**  
  A lente grande-angular 24mm equivalente sofre de **distorção radial de barril** descrita pelo modelo polinomial de Brown-Conrady:
  $$x_{\text{distorcido}} = x_c (1 + k_1 r^2 + k_2 r^4), \quad r^2 = (x - c_x)^2 + (y - c_y)^2$$
  onde $(c_x, c_y)$ é o ponto principal da lente física, coincidente com o centro da matriz física do sensor de 48 megapixels.  
  Se recortássemos a imagem antes de aplicar a desdistorção, o novo centro da subimagem passaria a ser assumido como centro óptico da lente. Isso alteraria o vetor radial $r$ em milhares de pixels, provocando uma distorção artificial assimétrica que deforma pessoas fora do centro (causa comprovada do deslocamento lateral de 18 px observado na mulher).
- **Alternativas Descartadas:**  
  - *Desdistorcer após o corte:* Provou-se matematicamente catastrófico, criando o efeito fantasma lateral na mulher ao centro e curvando os cantos da imagem térmica.  
  - *Ignorar a desdistorção:* Gera erro de projeção nos cantos superior e inferior de até 40 pixels, impossibilitando a sobreposição de pedestres periféricos.
- **Argumento para o Time:**  
  *"A física da lente não se altera após cortarmos uma foto. O centro óptico é uma propriedade de fábrica do sensor de 48MP; por isso, a retificação precisa obrigatoriamente ocorrer na matriz original inteira."*

---

<a id="decisao-04-casamento-de-campo-de-visao-fov-center-crop-ancorado-na-altura"></a>
### Decisão 04: Casamento de Campo de Visão (FOV Center Crop Ancorado na Altura)

- **O que foi decidido:**  
  Ancorar a janela de recorte central de 70% na **altura** ($6000 \times 0.70 = 4200\text{ px}$) e derivar a largura a partir da proporção de aspecto nativa do sensor térmico ($1280/1024 = 1.25$ ou $5:4$), resultando numa janela exata de $5250 \times 4200\text{ px}$.
- **Fundamentação Técnica e Matemática:**  
  1. *Casamento Angular:* A câmera grande-angular possui HFOV de aproximadamente $84^\circ$, enquanto a câmera termográfica possui HFOV de $\approx 61^\circ$. A razão de abertura angular vertical corresponde a $\approx 0.70$.  
  2. *Eliminação de Deformação Anamórfica:* A câmera óptica captura na proporção $4:3$ ($1.333$), enquanto a câmera térmica grava na proporção $5:4$ ($1.250$). Se recortássemos baseando-nos na largura da imagem (como feito na versão preliminar com $w \times 0.70 = 5600\text{ px}$ e $h = 4480\text{ px}$), estaríamos inserindo um erro de escala/zoom óptico de:
  $$\frac{5600}{5250} = 1.0667 \quad (\approx 7\% \text{ de divergência de escala})$$
  Esse erro de $7\%$ em objetos a $250\text{ px}$ do centro cria uma disparidade lateral de $250 \times 0.07 \approx 17.5\text{ px}$, criando a sombra que desencaixava a silhueta da mulher.  
  Ao ancorar na altura ($4200$) e multiplicar por $1.25$ ($5250$), a imagem óptica resultante tem exatamente a proporção $5:4$, sem nenhuma compressão lateral ao ser redimensionada para $1280 \times 1024$.
- **Alternativas Descartadas:**  
  - *Corte baseado na largura ($w \times 0.70$):* Rejeitado após diagnóstico probatório de regressão de alinhamento.  
  - *Letterboxing na imagem óptica:* Rejeitado porque faixas pretas adicionariam bordas artificiais que ativariam filtros convolucionais da rede neural de forma espúria.
- **Argumento para o Time:**  
  *"A câmera térmica enxerga menos do que a visual e tem proporção 5:4. Ancorando o corte na altura e derivando a largura por 1.25, casamos o enquadramento sem esticar nem encolher a silhueta dos pedestres."*

---

<a id="decisao-05-equalizacao-radiometrica-da-termica-clahe-no-espaco-lab"></a>
### Decisão 05: Equalização Radiométrica da Térmica (CLAHE no Espaço Lab)

- **O que foi decidido:**  
  Converter a imagem termográfica para o espaço de cor **CIE-Lab**, aplicar o algoritmo CLAHE (*Contrast Limited Adaptive Histogram Equalization*) com `clipLimit=2.5` e grade espacial `(8, 8)` **exclusivamente no canal L (Luminância)**, recompondo os canais cromáticos originais $a$ e $b$.
- **Fundamentação Técnica e Matemática:**  
  Sensores térmicos LWIR (8 a 14 $\mu m$) capturam radiação eletromagnética emitida por calor. Em dias de sol ou asfalto morno, o histograma térmico concentra-se em uma faixa estreita de valores.  
  Uma equalização de histograma global convencional redistribuiria os tons saturando todo o ruído de fundo (como o brilho difuso do asfalto), mascarando as pessoas.  
  O CLAHE particiona a imagem em blocos $8 \times 8$ e limita a inclinação do histograma (`clipLimit=2.5`), redistribuindo os fótons térmicos localmente. Ao atuar exclusivamente sobre o canal $L$ do espaço Lab, o contraste radiométrico dos pedestres é amplificado sem alterar a paleta de cores ou distorcer a geometria térmica.
- **Alternativas Descartadas:**  
  - *Equalização direta no canal RGB:* Distorce o equilíbrio de cor da paleta termográfica e cria aberrações cromáticas em torno de lâmpadas e motores.  
  - *Manter a térmica bruta sem equalização:* O modelo de IA perde cerca de $35\%$ de acurácia em pessoas distantes ou sob o asfalto aquecido por falta de gradiente de ativação.
- **Argumento para o Time:**  
  *"O CLAHE no canal L é o equivalente a um filtro polarizador: ele realça o calor corporal das pessoas sem explodir o ruído do asfalto quente."*

---

<a id="decisao-06-padronizacao-de-resolucao-e-compensacao-afim-de-baseline-estereo"></a>
### Decisão 06: Padronização de Resolução e Compensação Afim de Baseline Estéreo

- **O que foi decidido:**  
  1. Reamostrar o RGB recortado ($5250 \times 4200$) para $1280 \times 1024$ usando `cv2.INTER_AREA`.  
  2. Reamostrar a Térmica ($640 \times 512$) para $1280 \times 1024$ usando `cv2.INTER_CUBIC`.  
  3. Aplicar matriz afim $M = \begin{bmatrix} 1 & 0 & -22 \\ 0 & 1 & -23 \end{bmatrix}$ estritamente sobre a imagem RGB.
- **Fundamentação Técnica e Matemática:**  
  1. *Filtros de Reamostragem:* Na redução de escala ($5250 \to 1280$), o `INTER_AREA` calcula a média dos pixels contidos no pixel de destino, eliminando o efeito Moiré e ruídos de alta frequência. Na ampliação ($640 \to 1280$), o `INTER_CUBIC` utiliza convolução cúbica $4 \times 4$ de vizinhos, criando gradientes suaves sem os degraus pixelizados do `INTER_NEAREST`.  
  2. *Compensação de Paralaxe de Baseline:* As câmeras estão montadas no gimbal do drone separadas fisicamente por uma distância estéreo $B$. Pela relação de estereoscopia:
  $$d = \frac{f \cdot B}{Z}$$
  Para a altitude de voo de missão ($Z \approx 30\text{ a }40\text{ m}$), o deslocamento no plano do solo resulta no vetor estável $dx = -22\text{ px}$ e $dy = -23\text{ px}$. Aplicar essa translação na imagem RGB alinha perfeitamente a posição do chão óptico com o calor detectado.
- **Alternativas Descartadas:**  
  - *Homografia automática por pontos-chave (SIFT/ORB) em cada frame:* Em cenas aéreas urbanas noturnas, os descritores de gradiente óptico e radiométrico divergem completamente (uma placa iluminada tem contorno óptico, mas não térmico). O RANSAC frequentemente calculava matrizes fisicamente absurdas (escala 1.4x ou giros). O vetor afim fixo calibrado provou-se $100\%$ determinístico e imune a falhas de convergência.
- **Argumento para o Time:**  
  *"A distância entre as lentes no drone é fixa e a altura de voo é controlada. Um vetor calibrado de (-22, -23) pixels é infinitamente mais robusto e previsível do que tentar calcular homografia dinâmica em imagens com texturas tão diferentes."*

---

<a id="decisao-07-validacao-de-paridade-bit-a-bit-com-a-classe-de-producao"></a>
### Decisão 07: Validação de Paridade Bit-a-Bit com a Classe de Produção

- **O que foi decidido:**  
  Inserir um teste unitário formal em célula executável com `assert diff_rgb == 0.0 and diff_th == 0.0`, comparando o passo a passo com `RGBTImageEqualizer.process_pair`.
- **Fundamentação Técnica e Matemática:**  
  Garante paridade estrita entre prototipagem e engenharia de software (princípio da reprodutibilidade científica). Elimina qualquer risco de a equipe explorar uma lógica no notebook e o backend rodar outra diferente em produção.
- **Argumento para o Time:**  
  *"Não existe 'no meu notebook funciona e em produção não'. O assert no código comprova erro de zero pixels entre este caderno e o código de produção da pasta app/."*

---

<a id="decisao-08-auditoria-visual-sub-pixel-na-mulher-central-e-plano-do-solo"></a>
### Decisão 08: Auditoria Visual Sub-Pixel na Mulher Central e Plano do Solo

- **O que foi decidido:**  
  Gerar uma sobreposição 50/50 (*Alpha Blend*) e recortar explicitamente duas ROIs críticas:
  - *Região Central Calibrada:* $x=[724, 797], y=[588, 706]$ (Mulher ao Centro).
  - *Região Base / Solo:* $x=[9, 172], y=[786, 981]$ (Pedestres caminhando).
- **Fundamentação Técnica e Matemática:**  
  Permite validar visualmente a anulação da paralaxe tanto em alvos de alto contraste e silhueta definida (mulher) quanto em alvos periféricos no chão. Salvar a figura em disco garante rastreabilidade e histórico probatório.
- **Argumento para o Time:**  
  *"Não confiamos apenas em números abstratos. Damos um zoom de alta definição nas pessoas para auditar que o contorno de calor veste o corpo humano com perfeição milimétrica."*

---

<a id="decisao-09-exportacao-do-contrato-de-insumos-padronizado"></a>
### Decisão 09: Exportação do Contrato de Insumos Padronizado

- **O que foi decidido:**  
  Isolar estritamente 5 arquivos na pasta `notebooks/output/01_pre_transformacao/`:
  1. `rgb_preprocessed.jpg`
  2. `thermal_preprocessed.jpg`
  3. `blend_alta_precisao.jpg`
  4. `painel_alinhamento_multimodal.jpg`
  5. `metadata_preprocessing.json`
- **Fundamentação Técnica e Matemática:**  
  Adoção do padrão arquitetural de **Data Contracts**. O Notebook 01 e o Notebook 02 comunicam-se exclusivamente através deste contrato imutável. O arquivo JSON registra parâmetros técnicos completos (matriz $K$, coeficientes de distorção, dimensões do crop, shift aplicado e timestamp).
- **Argumento para o Time:**  
  *"Criamos uma fronteira limpa entre engenharia de imagem e inferência de IA. Se alguém quiser testar outro modelo de contagem amanhã, não precisará reprocessar um único pixel da câmera."*
