# Repositório de Ground Truth (Anotações Manuais) • DEF-RGBTCC

Este diretório centraliza os registros de contagem manual (*Ground Truth* real) para validação dos modelos neurais de contagem de multidões em imagens aéreas de drones.

Como a anotação pontual de cabeças é realizada visualmente sobre uma imagem específica (normalmente o sensor óptico RGB Wide de altíssima resolução), **cada diretório é nomeado diretamente com o identificador da imagem contada** (ex: `DJI_0789_W`).

---

## 1. Estrutura de Diretórios por Imagem Contada

```text
data/ground_truth/
├── README.md
│
├── DJI_0789_W/                                     # Imagem contada (532 pessoas reais)
│   ├── checkpoint_anotacao.json                    # Checkpoint em tempo real da anotação
│   ├── pontos_ground_truth.json                    # Pontos e metadados no espaço RAW (8000x6000 px)
│   ├── pontos_ground_truth.csv                     # Tabela CSV (id, x, y) no espaço RAW
│   ├── ground_truth_aligned_1280x1024.json         # Coordenadas projetadas no espaço de inferência (1280x1024 px)
│   ├── rgb_anotada_ground_truth_DJI_0789_W.jpg     # Imagem original com pontos marcados (auditoria visual)
│   └── metadados.json                              # Ficha técnica da imagem contada
│
└── ... (novas imagens contadas adicionadas dinamicamente)
```

---

## 2. Contrato de Arquivos de Cada Imagem Contada

Dentro da pasta de cada imagem, os seguintes arquivos padronizados são mantidos:

| Arquivo | Descrição | Espaço de Coordenadas | Finalidade |
| :--- | :--- | :--- | :--- |
| **`checkpoint_anotacao.json`** | Estado de progresso salvo automaticamente ou via `Ctrl+S` | RAW ($8000 \times 6000$) | Permite pausar e retomar a anotação a qualquer momento sem perda de dados |
| **`pontos_ground_truth.json`** | Lista final validada com metadados completos | RAW ($8000 \times 6000$) | Registro mestre permanente da contagem manual |
| **`pontos_ground_truth.csv`** | Tabela tabular com colunas `id, x, y` | RAW ($8000 \times 6000$) | Interoperabilidade com Pandas, R e ferramentas externas |
| **`ground_truth_aligned_1280x1024.json`** | Pontos mapeados para a resolução de inferência | Alinhado ($1280 \times 1024$) | Consumido pelos notebooks de inferência para gerar mapa gaussiano e calcular MSE/RMSE/NAE |
| **`rgb_anotada_ground_truth_*.jpg`** | Imagem de alta resolução com marcações em vermelho/amarelo | RAW ($8000 \times 6000$) | Auditoria visual humana da integridade da anotação |
| **`metadados.json`** | Ficha técnica da cena e resumo estatístico | - | Informações de drone, altitude, horário e total real |

---

## 3. Como Anotar uma Nova Imagem

Para iniciar ou continuar a contagem manual de qualquer imagem, execute o anotador interativo:

```bash
./notebooks/.venv/bin/python scripts/anotar_pontos.py \
    --imagem notebooks/DEF-rgbtcc/input/DJI_0777_W.JPG
```

O script detecta o nome da imagem (`DJI_0777_W`) e cria/utiliza automaticamente a pasta:  
`data/ground_truth/DJI_0777_W/`

### Controles Principais do Anotador:
* **Rolar Mouse (Scroll):** Zoom In / Zoom Out centrado no cursor do mouse (até 12x de ampliação para identificar cada cabeça individual).
* **Botão do Meio ou Segurar [Espaço] + Botão Esquerdo:** Arrastar (Pan) suavemente pela imagem 8000x6000.
* **Teclas W, A, S, D ou Setas:** Mover o viewport pela cena.
* **Botão Esquerdo:** Marcar cabeça (ponto).
* **Botão Direito / [Ctrl + Z]:** Desfazer último ponto.
* **[Ctrl + S] ou Botão [Salvar]:** Gravar checkpoint imediato.
* **Tecla [F] ou Botão [Finalizar]:** Finalizar a sessão e gerar todos os entregáveis (CSV, JSONs, projeção 1280x1024 e Imagem anotada).
