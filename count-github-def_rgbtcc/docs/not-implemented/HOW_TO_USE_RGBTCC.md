# Especificação Técnica de Imagens e Guia de Uso: Modelo RGBTCC (BMVC 2022)

Este documento estabelece as **especificações técnicas exatas** de imagens, formatos, dimensões, canais, estatísticas e calibrações necessárias para alimentar o modelo multimodal **RGBTCC** (*RGB-Thermal Crowd Counting Based on Transformer*, Liu et al., BMVC 2022), garantindo a máxima acurácia na identificação e contagem de pessoas.

---

## 1. Visão Geral da Arquitetura e do Mecanismo de Entrada

O modelo RGBTCC baseia-se em um paradigma de **Regressão de Mapa de Densidade Espacial** (*Density Map Estimation*) multimodal:

```
                          ┌───────────────────────────┐
                          │   PAR MULTIMODAL DE FATO  │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       [Stream 1: Óptico RGB]                       [Stream 2: Térmico T]
       • Backbone: PVTv2-B3                         • Backbone: PVTv2-B3
       • 4 Estágios Hierárquicos                   • 4 Estágios Hierárquicos
       • Textura, cor, contornos                    • Assinatura calórica infravermelha
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                      [Count-guided Multi-Scale Attention (MSA)]
                                        ▼
                 [Multi-Scale Deformable Transformer Decoder]
                                        ▼
                     [Mapa de Densidade 2D Reconstruído]
                           Contagem Total = ∑ D(x, y)
```

> [!IMPORTANT]
> **O papel da Imagem de Blend (Overlay):**  
> A rede neural **NÃO consome a imagem de blend/overlay como entrada direta**. A arquitetura possui duas portas de entrada independentes (`inputs[0] = RGB` e `inputs[1] = Térmico`).  
> A imagem `blend_alta_precisao_app_final.jpg` (gerada no ADR 007) é um artefato de **auditoria de pré-processamento** (para verificar se as silhuetas estão alinhadas) e de **projeção visual pós-inferência** (para renderizar o mapa de calor com transparência sobreposto à cena).

---

## 2. Ficha Técnica Completa de Entrada (Cheat-Sheet)

| Parâmetro | Especificação Estrita | Justificativa Técnica / Código de Origem |
| :--- | :--- | :--- |
| **Composição da Entrada** | **Par obrigatório**: 1 imagem RGB + 1 imagem Térmica | Arquitetura Dual-Stream com backbones paralelos (`pvt_v2_b3`). |
| **Formato de Arquivo** | `.jpg` ou `.png` | Imagens 8-bit sem compressão destrutiva extrema. |
| **Profundidade de Cor** | 8 bits por canal (`uint8`, $[0, 255]$) | Compatível com leitura padrão OpenCV (`cv2.imread`). |
| **Número de Canais (RGB)** | **3 canais** (R, G, B) | O PVTv2 óptico espera `[B, 3, H, W]`. |
| **Número de Canais (Térmica)** | **3 canais** (R, G, B) | O PVTv2 térmico também possui camada convolucional inicial com `in_channels=3`. Se a câmera fornecer 1 canal cinza (LWIR), deve-se triplicar o canal ($C_1=C_2=C_3$). |
| **Ordem dos Canais** | **RGB** | OpenCV lê como BGR; o repositório inverte explicitamente via `cv2.imread(path)[..., ::-1]`. |
| **Resolução Recomendada (Tiling)** | **$672 \times 448$ px** (Aspect Ratio 3:2) | Múltiplo exato de 224 ($3 \times 224$ na largura, $2 \times 224$ na altura) para fatiamento em 6 patches. |
| **Resolução Nativa (Drone / DJI)** | **$640 \times 512$ px** (Aspect Ratio 5:4) | Múltiplo exato de 32 ($640/32 = 20$, $512/32 = 16$). Pode ser inferida direto se for compatível com divisibilidade por 32. |
| **Restrição Espacial Absoluta** | **$H \pmod{32} == 0$ e $W \pmod{32} == 0$** | Os 4 estágios do PVTv2 reduzem até $1/32$. Se não for divisível por 32, há erro de dimensão na camada linear `pre_out`. |
| **Normalização Visual (RGB)** | $\mu = [0.407, 0.389, 0.396]$<br>$\sigma = [0.241, 0.246, 0.242]$ | Médias e desvios empíricos do benchmark RGBT-CC (`datasets/crowd.py:66-67`). |
| **Normalização Térmica (T)** | $\mu = [0.492, 0.168, 0.430]$<br>$\sigma = [0.317, 0.174, 0.191]$ | Estatísticas originais do sensor térmico do dataset (`datasets/crowd.py:72-73`). Não usar ImageNet genérico. |
| **Tolerância de Desalinhamento** | **$< 3$ pixels (Erro Subpixel)** | A atenção cruzada correlaciona tokens na mesma coordenada espacial. Deslocamentos geram dupla contagem ou supressão. |

---

## 3. Detalhamento Técnico das Características das Imagens

### 3.1 Modalidade Visual: Imagem RGB
* **Finalidade**: Extrair características de alta frequência espacial — bordas dos ombros, textura do cabelo, contorno de pedestres e detalhes de vestuário.
* **Pré-tratamento Requerido**:
  - Correção de distorção radial de lente (undistort) para garantir geometria retilínea (ADR 001).
  - Casamento de FOV (*Field of View*) em relação à câmera térmica.
* **Espaço de Cores**: RGB padrão sRGB, valores entre 0 e 255.

---

### 3.2 Modalidade Térmica: Imagem Infravermelha (LWIR)
* **Finalidade**: Isolar a radiância infravermelha térmica emitida pelo corpo humano (espectro de onda longa, $8\,\mu\text{m} - 14\,\mu\text{m}$), fornecendo imunidade total a sombras de árvores, escuridão noturna ou oclusões visuais sutis.
* **Conversão para 3 Canais**:
  - Sensores térmicos puros geram dados monocromáticos (1 canal).
  - O modelo **requer obrigatoriamente 3 canais**. A conversão recomendada é:
    ```python
    # Opção A: Replicação direta em 3 canais cinza
    thermal_3c = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2RGB)
    
    # Opção B: Pseudo-cor térmica com equalização CLAHE (método adotado em data/gold/manually)
    # Garante alto gradiente térmico entre a cabeça humana e o piso
    ```
* **Contraste Radiométrico**: Aplicar realce adaptativo de contraste local (**CLAHE**) com `clipLimit=3.0` no espaço de luminância $L^*$ (LAB). Isso evita que o calor residual do asfalto ofusque as pessoas.

---

### 3.3 Co-registro Espacial e Compensação de Paralaxe
* Devido à separação física entre as lentes óptica e térmica em drones/câmeras duplas (baseline de $1.5\text{ cm} - 3.5\text{ cm}$), ocorre paralaxe óptica.
* O modelo espera que **o pixel $(x, y)$ da imagem RGB corresponda exatamente ao pixel $(x, y)$ da imagem térmica no plano do solo**.
* Se houver deslocamento:
  - O token térmico da pessoa cai sobre o fundo na imagem RGB, anulando a atenção cruzada do módulo MSA.
  - A imagem tratada em `app/data/gold/manually/` já incorpora a matriz de transformação afim de solo (**ADR 007**), cumprindo esse requisito com excelência.

---

## 4. Estratégias de Resolução: Tiling ($224 \times 224$) vs. Resolução Completa

O projeto `liuzywen-RGBTCC` suporta dois modos operacionais de alimentação:

### Modo A: Fatiamento em Grade (Padrão Original do Autor — Recomendado)
1. Redimensionar o par de imagens tratadas para **$672 \times 448$ px**.
2. Fatiar cada imagem em uma grade de $3 \times 2 = 6$ recortes (*crops*) de **$224 \times 224$ px**:
   $$\text{Grade: } 3 \text{ colunas} \times 2 \text{ linhas} = 6 \text{ patches}$$
3. Montar o lote de tensores:
   $$\text{Shape RGB: } [6, 3, 224, 224] \qquad \text{Shape Térmico: } [6, 3, 224, 224]$$
4. A rede processa os 6 patches em paralelo na GPU.
5. Cada patch produz um mini-mapa de densidade $[1, 28, 28]$.
6. Os 6 mapas são concatenados no formato $2 \times 3$, reconstruindo o mapa final da cena com resolução **$56 \times 84$ px**.
7. **Vantagem**: Máxima fidelidade com a distribuição de escala onde a rede foi treinada.

### Modo B: Imagem Completa Nativa de Drone ($640 \times 512$ px)
1. Como $640 / 32 = 20$ e $512 / 32 = 16$, as dimensões nativas de 5:4 são matematicamente compatíveis com o PVTv2.
2. A imagem entra como lote unitário $[1, 3, 512, 640]$.
3. O mapa de densidade resultante possui resolução de $64 \times 80$ px (ou interpolado para $512 \times 640$).
4. **Vantagem**: Não necessita corte ou remontagem de bordas de patches.

---

## 5. Pipeline Python de Pré-processamento Padronizado

Abaixo está o código de referência que implementa rigorosamente todas as especificações acima:

```python
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

def load_and_preprocess_pair(rgb_path: str, thermal_path: str, target_size=(448, 672), device="cuda"):
    """Carrega e prepara o par multimodal conforme as especificações estritas do RGBTCC.
    
    Args:
        rgb_path: Caminho da imagem RGB tratada (ex: data/gold/manually/rgb_final.jpg).
        thermal_path: Caminho da imagem térmica tratada (ex: data/gold/manually/thermal_final.jpg).
        target_size: Tupla (Altura, Largura) - default (448, 672) para grade 3x2 de 224x224.
        device: 'cuda' ou 'cpu'.
        
    Returns:
        batch_rgb: Tensor [6, 3, 224, 224] pronto para inferência.
        batch_th: Tensor [6, 3, 224, 224] pronto para inferência.
    """
    assert os.path.exists(rgb_path), f"RGB não encontrado: {rgb_path}"
    assert os.path.exists(thermal_path), f"Térmica não encontrada: {thermal_path}"
    
    # 1. Leitura com OpenCV e inversão para RGB
    img_rgb = cv2.imread(rgb_path)[..., ::-1].copy()
    img_th = cv2.imread(thermal_path)[..., ::-1].copy()
    
    target_h, target_w = target_size
    assert target_h % 224 == 0 and target_w % 224 == 0, "Dimensões devem ser múltiplos de 224!"
    
    # 2. Redimensionamento
    img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_th = cv2.resize(img_th, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 3. Normalização estatística estrita do RGBT-CC
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.407, 0.389, 0.396], std=[0.241, 0.246, 0.242])
    ])
    transform_th = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191])
    ])
    
    t_rgb = transform_rgb(img_rgb)
    t_th = transform_th(img_th)
    
    # 4. Fatiamento em patches de 224x224
    m = target_w // 224
    n = target_h // 224
    
    crops_rgb, crops_th = [], []
    for i in range(m):
        for j in range(n):
            c_rgb = t_rgb[:, j*224:(j+1)*224, i*224:(i+1)*224].unsqueeze(0)
            c_th = t_th[:, j*224:(j+1)*224, i*224:(i+1)*224].unsqueeze(0)
            crops_rgb.append(c_rgb)
            crops_th.append(c_th)
            
    batch_rgb = torch.cat(crops_rgb, dim=0).to(device)
    batch_th = torch.cat(crops_th, dim=0).to(device)
    
    return batch_rgb, batch_th
```

---

## 6. Checklist de Qualidade para Novos Lotes de Imagens

Antes de alimentar qualquer nova imagem no modelo para contagem em produção, verifique:

- [ ] **Paridade 1:1**: Existe exatamente uma imagem visual e uma térmica correspondente capturadas no mesmo instante (timestamp idêntico).
- [ ] **Co-registro Geométrico**: Pedestres no chão estão alinhados entre as duas imagens com tolerância $< 3$ pixels.
- [ ] **3 Canais Válidos**: A imagem térmica possui 3 canais e não está salva como escala de cinza mono-canal não convertida.
- [ ] **Resolução Ajustada**: Imagens pré-processadas para $672 \times 448$ (ou $640 \times 512$).
- [ ] **Sem Letterbox Preto no Fatiamento**: Não utilizar barras pretas que cruzem patches de $224 \times 224$, pois bordas artificiais de alto contraste geram falsos positivos de cabeça.

