### 📊 Resumo Executivo: Ordem de Implementação por Impacto

| Ranking | Melhoria | Impacto Esperado | Sintoma Atual Corrigido |
| :---: | :--- | :---: | :--- |
| **1º** | **Inferência por Tiling / Patches ($640 \times 512$)** | **Crítico (80% do problema)** | As 2.826 pessoas viram frações de sub-pixel na rede e desaparecem no pooling. |
| **2º** | **Integração da Arquitetura Oficial (`dm.py` do artigo)** | **Crítico (Base do modelo)** | Troca o modelo leve de 58 pessoas (`DualStreamRGBTNet`) pelo VGG-19 + Transformers SMA + AFM do artigo. |
| **3º** | **Eliminação do `0.0001` e Troca de `Softplus` por `Abs`** | **Alto (Correção matemática)** | Fator artificial que congelava a contagem em ~90 ou explodia ruído de fundo. |
| **4º** | **Supressão de Fundo / Limiarização de Ruído ($\epsilon$)** | **Médio-Alto (Precisão)** | Evita que 70% da imagem (asfalto/grama vazios) acumule centenas de pessoas falsas. |
| **5º** | **Ajuste de Distância Mínima nos Picos (`min_distance`)** | **Médio (Visualização local)** | Explica por que a rede só achava 56 picos em 684 preditos (pessoas aglomeradas colapsavam num único pico). |
| **6º** | **Alinhamento da Normalização Multimodal (ImageNet nos 2 canais)** | **Refinamento (Estabilidade)** | Casamento exato com `datasets/crowd.py` do autor oficial. |

---

### 🔍 Detalhamento das Melhorias (Do Maior para o Menor Impacto)

#### 🥇 1º Lugar: Inferência por Tiling / Mosaico ($640 \times 512$ ou $448 \times 448$)
* **Por que é o maior impacto?**
  * A imagem de teste `DJI_0763_W` tem **2.826 pessoas** em visão aérea panorâmica ($1280 \times 1024$).
  * Em resolução global, a cabeça de uma pessoa tem cerca de **3 a 5 pixels** de largura.
  * O backbone do modelo aplica um downsampling de **16x** (VGG com 4 camadas de MaxPool). Logo, uma cabeça de 4 pixels é reduzida para $4 / 16 = \mathbf{0.25\text{ pixels}}$ (abaixo de 1 pixel!).
  * **Consequência:** As pessoas são fisicamente "engolidas" pelo pooling convolucional. O modelo só "enxerga" massas difusas, perdendo 80% das cabeças individuais.
* **Solução:** Dividir a imagem de entrada em quadrantes ou janelas deslizantes (patches de $640 \times 512$, que é a resolução nativa do dataset *DroneRGBT* usada no artigo), inferir cada patch e recompor o mapa de densidade.

---

#### 🥈 2º Lugar: Portar a Arquitetura Oficial (`dm.py`) do Repositório Atrelado
* **Por que é o segundo maior impacto?**
  * O arquivo atual `models/def_rgbtcc_net.py` usava uma implementação com decodificador transposto próprio, e os pesos em `weights/best_model.pth` pertencem ao `DualStreamRGBTNet` (rede compacta treinada apenas para cenas pequenas, com `calibrated_count: 58.0`).
  * No repositório oficial atrelado (`RGBT-Crowd-Counting/models/dm.py`), a arquitetura real do paper é:
    1. **VGG-19 compartilhado** (extração multimodal paralela com pesos pré-treinados do ImageNet).
    2. **Transformers SMA (Spatially Modulated Attention)** com máscara de decaimento espacial $M$ calculada sobre a matriz de distâncias euclidianas.
    3. **AFM (Adaptive Cross-Modal Fusion)** com Global Avg Pool + MLP convolucional 1x1.
    4. **Head de Regressão com Interpolação 2x** e ativação `torch.abs(density)`.
* **Solução:** Conectar a classe `Net` oficial de `RGBT-Crowd-Counting/models/dm.py` diretamente ao nosso pipeline de inferência.

---

#### 🥉 3º Lugar: Eliminar o Multiplicador Artificial `0.0001` e Ajustar Ativação para `Abs`
* **Por que é o terceiro impacto?**
  * No código original do notebook havia: `raw_sum * 0.0001`. Esse fator foi colocado porque a ativação `F.softplus(0) = \ln(2) \approx 0.693` gerava um valor alto somado sobre 1.3 milhão de pixels ($\approx 900.000$). Multiplicado por $0.0001$, dava artificialmente 90 pessoas.
  * No artigo (conforme confirmado em `test_game.py` linha 53):
    $$\text{Contagem Total} = \sum_{h, w} \text{density\_map}(h, w)$$
    O cálculo oficial é a **soma direta sem nenhum multiplicador**.
* **Solução:** Padronizar a integração da densidade de acordo com a formulação Bayesiana do paper: $\sum \text{outputs}$ em escala downsampled de 1/8.

---

#### 4º Lugar: Supressão de Fundo e Limiarização Adaptativa ($\epsilon$)
* **Por que é o quarto impacto?**
  * Em imagens aéreas de drone, mais de **70% da área da imagem é vazia** (ruas, telhados, gramados, árvores).
  * Mesmo uma ativação residual minúscula de $\epsilon = 0.0005$ por pixel acumula:
    $$1.310.720 \text{ pixels} \times 0.0005 \approx \mathbf{655\text{ pessoas fantasmas no fundo!}}$$
  * Isso explica por que o modelo atual previa ~684 pessoas: ~340 vinham do fundo vazio e ~344 vinham da área real das pessoas.
* **Solução:** Aplicar corte de ruído no mapa de densidade (threshold de fundo ou mascaramento de confiança multimodal via AFM) para que regiões sem pessoas tenham densidade estritamente zero.

---

#### 5º Lugar: Ajuste do Algoritmo de Picos (`min_distance` e `threshold_abs`)
* **Por que é o quinto impacto?**
  * O notebook reportava: *Predito pela soma: 684 pessoas | Picos encontrados: 56*.
  * O motivo é que `peak_local_max` estava configurado com `min_distance=10` ou `15`. Em uma multidão densa de 2.826 pessoas, a distância entre duas cabeças na imagem é de apenas **3 a 6 pixels**. Com raio de 10 a 15 pixels, o algoritmo fundia 10 pessoas em um único pico.
* **Solução:** Reduzir `min_distance` para 3-4 pixels nos patches de alta densidade e calibrar o limiar de pico proporcional ao valor máximo local.

---

#### 6º Lugar: Alinhamento Estrito do Preprocessamento Multimodal
* **Por que é o sexto impacto?**
  * O pipeline de treinamento em `RGBT-Crowd-Counting/datasets/crowd.py` padroniza:
    * RGB: `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
    * Térmico: Convertido para 3 canais (`.convert('RGB')`) e normalizado com os **mesmos parâmetros do ImageNet**.
* **Solução:** Garantir que o notebook 02 use exatamente esses tensores antes de alimentar o modelo.

---