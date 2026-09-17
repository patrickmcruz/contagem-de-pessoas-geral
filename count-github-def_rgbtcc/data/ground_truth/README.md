# Repositório de Ground Truth (Anotações Manuais) • DEF-RGBTCC

Este diretório centraliza os registros de contagem manual (*Ground Truth* real) para validação dos modelos neurais de contagem de multidões em imagens aéreas de drones.

Como a anotação pontual de cabeças é realizada visualmente sobre uma imagem específica (normalmente o sensor óptico RGB Wide de altíssima resolução), **cada diretório é nomeado diretamente com o identificador da imagem contada** (ex: `DJI_0789_W`).

---

## 1. Estrutura de Diretórios por Imagem Contada

```text
data/ground_truth/
├── README.md
│
├── DJI_0763_W/                                     # Imagem contada (2.826 pessoas reais)
├── DJI_0765_W/                                     # Imagem contada (1.082 pessoas reais)
├── DJI_0767_W/                                     # Imagem contada (1.842 pessoas reais)
├── DJI_0779_W/                                     # Imagem contada (740 pessoas reais)
├── DJI_0789_W/                                     # Imagem contada (532 pessoas reais)
│   ├── checkpoint_anotacao.json                    # Checkpoint em tempo real da anotação
│   ├── pontos_ground_truth.json                    # Pontos e metadados no espaço nativo
│   ├── pontos_ground_truth.csv                     # Tabela CSV (id, x, y) no espaço nativo
│   ├── ground_truth_aligned_1280x1024.json         # Coordenadas projetadas no espaço de inferência (1280x1024 px)
│   ├── rgb_anotada_ground_truth_*.jpg              # Imagem original com pontos marcados (auditoria visual)
│   └── metadados.json                              # Ficha técnica da imagem contada
│
└── ... (novas imagens contadas adicionadas dinamicamente)
```

### Catálogo de Cenas Ground Truth Disponíveis

| Cena / Stem | Origem | Resolução Nativa | Resolução Alinhada | Total Pessoas Reais |
| :--- | :--- | :--- | :--- | :--- |
| **`DJI_0763_W`** | `data/manual_counting_check/` | $1536 \times 1152$ | $1280 \times 1024$ | **2.826** cabeças |
| **`DJI_0765_W`** | `data/manual_counting_check/` | $1536 \times 1152$ | $1280 \times 1024$ | **1.082** cabeças |
| **`DJI_0767_W`** | `data/manual_counting_check/` | $1536 \times 1152$ | $1280 \times 1024$ | **1.842** cabeças |
| **`DJI_0779_W`** | `data/manual_counting_check/` | $1536 \times 1152$ | $1280 \times 1024$ | **740** cabeças |
| **`DJI_0789_W`** | `notebooks/DEF-rgbtcc/input/` | $8000 \times 6000$ | $1280 \times 1024$ | **532** cabeças |
| **Total Acumulado** | **5 cenas** | - | - | **7.022 pessoas reais** |


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
    --imagem data/input/DJI_0789_W.JPG
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

---

## 4. Como Importar Lotes de Contagem Manual (`data/manual_counting_check/`)

Se você já possui anotações manuais pontuais em arquivos de texto (onde cada linha do `.txt` contém as coordenadas espaciais `X Y` correspondentes à imagem `.JPG`), utilize o importador automatizado:

```bash
./notebooks/.venv/bin/python scripts/importar_manual_ground_truth.py
```

Parâmetros opcionais:
- `--input-dir`: Diretório com os pares imagem e `.txt` (padrão: `data/manual_counting_check`).
- `--output-dir`: Diretório raiz de Ground Truth (padrão: `data/ground_truth`).
- `--largura-alinhada`: Largura do espaço de inferência para projeção (padrão: `1280`).
- `--altura-alinhada`: Altura do espaço de inferência para projeção (padrão: `1024`).

