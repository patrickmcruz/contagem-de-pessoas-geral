# YOLO Head Counting - Aplicação Isolada (/app)

Esta pasta contém o subprojeto **`app`**, que implementa de forma 100% isolada e autônoma o pipeline de detecção e contagem de cabeças humanas. 

---

## ⚙️ Guia de Instalação e Configuração (Primeiro Acesso)

Siga o passo a passo abaixo para clonar o repositório, configurar o ambiente virtual e rodar a aplicação em sua máquina local.

### 1. Clonar o Repositório
Abra o terminal do seu sistema operacional e execute o comando de clone:
```bash
git clone https://github.com/patrickmcruz/yolo26-notebook.git
cd yolo26-notebook/app
```

### 2. Criar o Ambiente Virtual (`.venv`)
Crie um ambiente Python isolado para evitar conflitos de dependências com seu sistema global. Certifique-se de usar o Python 3.8+ (recomendado 3.10 ou superior).

* **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  ```
* **Windows (PowerShell ou CMD)**:
  ```powershell
  python -m venv .venv
  ```

### 3. Ativar o Ambiente Virtual
Ative o ambiente virtual recém-criado:

* **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```
* **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

*Nota: Uma vez ativado, o prompt do terminal exibirá o prefixo `(.venv)`.*

### 4. Instalar as Dependências
Com o `.venv` ativado, instale os pacotes necessários especificados no arquivo `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Nota: O instalador do PyTorch embutido no `requirements.txt` está configurado para baixar automaticamente os drivers necessários do PyTorch 2.6.0 com suporte a CUDA 12.4 para aceleração GPU NVIDIA. Caso seu sistema não possua placa NVIDIA dedicada, o PyTorch executará em modo de fallback em CPU.*

---

## 🚀 Como Executar

### Executar via Terminal (CLI)
Você pode rodar o processador de vídeo direto no terminal especificando um arquivo de configuração `.yaml`:

* **Cenário Diurno (muralha-dia.mp4)**:
  ```bash
  python run.py --config data_day.yaml
  ```

* **Cenário Noturno (muralha-noite.mp4)**:
  ```bash
  python run.py --config data_night.yaml
  ```

Após a finalização, os resultados de saída estarão organizados nas respectivas subpastas dentro de `output/` (ex: `output/day/` e `output/night/`):
* **Vídeo Anotado**: `output/day/muralha-dia_annotated.mp4` (vídeo renderizado com as caixas de detecção).
* **Série Temporal (CSV)**: `output/day/frame_counts.csv` (contagem frame a frame para análise).
* **Relatório JSON**: `output/day/summary.json` (estatísticas agregadas e velocidade média em FPS).
* **Snapshots de Auditoria**: `output/day/snapshots/` (capturas periódicas da tela).

### Executar via Jupyter Notebook
Caso queira realizar execuções visuais e plotagem de gráficos dinâmicos:
1. Abra a pasta raiz do projeto em seu editor (ex: VSCode).
2. Abra o arquivo [`test-setup-11-heads-counting-enhanced.ipynb`](../notebooks/test-setup-11-heads-counting-enhanced.ipynb).
3. Selecione o kernel do ambiente virtual `.venv` (localizado em `app/.venv`) no canto superior direito do editor.
4. Escolha o arquivo de configuração desejado na primeira célula (`../app/data_day.yaml` ou `../app/data_night.yaml`) e execute as células sequencialmente.

---

## 🧪 Rodar os Testes Unitários

Para validar a integridade dos módulos internos e caminhos de arquivos de forma automatizada (sem depender de placa de vídeo física ativa no momento):
```bash
python -m pytest tests/
```

---

## 🏗️ Estrutura da Aplicação Isolada

```
count-github-yolo-01/app/
├── docs/                              # Documentação de arquitetura e engenharia
│   └── TECHNICAL_GUIDE.md             # Guia técnico detalhado de escolhas de engenharia e design
│
├── head_counting/                     # Pacote Python contendo a lógica central
│   ├── __init__.py                    # Interface de exports da biblioteca
│   ├── config.py                      # Parser dataclass tipado e resolução de caminhos locais
│   ├── model.py                       # Setup CUDA, compilação de pesos YOLO e inferência batch
│   ├── video.py                       # Threads de leitura e anotação/gravação de vídeo
│   └── pipeline.py                    # Orquestração de ponta a ponta do processo
│
├── input/
│   └── videos/                        # Vídeos originais para processamento
│       ├── muralha-dia.mp4           # Vídeo Diurno
│       └── muralha-noite.mp4          # Vídeo Noturno
│
├── weights/
│   └── best.pt                        # Pesos do modelo pré-treinado YOLO (especializado em cabeças)
│
├── tests/                             # Suíte de testes unitários
│   ├── test_config.py                 # Valida o carregamento e resolução do config
│   └── test_pipeline.py               # Mocka o processamento para validar o loop principal
│
├── data_day.yaml                      # Configuração para o cenário Diurno
├── data_night.yaml                    # Configuração para o cenário Noturno
├── run.py                             # CLI Runner executável
├── requirements.txt                   # Arquivo de requerimentos isolado para a aplicação
└── README.md                          # Este manual de instrução
```

---

## 📖 Guia de Engenharia e Arquitetura

Para obter um detalhamento minucioso de todas as decisões técnicas adotadas no projeto (como o padrão Producer-Consumer com filas thread-safe, tolerância a falhas via Adaptive Batch OOM de divisão e conquista, otimizações de VRAM e GPU RTX, e fundamentação matemática das métricas operacionais), consulte o arquivo central de documentação:

* **[TECHNICAL_GUIDE.md](count-github-yolo-01/app/docs/TECHNICAL_GUIDE.md)**

---

## 📚 Referências e Créditos de Modelos

Os pesos utilizados neste subprojeto (`best.pt` / `best.engine`) baseiam-se no modelo Yolov8n pré-treinado e disponibilizado publicamente pela comunidade de código aberto:

* **Modelo Original**: [irail-crowd-counting-yolov8n](https://huggingface.co/AmineSam/irail-crowd-counting-yolov8n)
* **Autor/Mantenedor**: [AmineSam (HuggingFace)](https://huggingface.co/AmineSam)
* **Descrição**: Um modelo especializado baseado na arquitetura Ultralytics YOLOv8n, otimizado para detecção densa de cabeças e contagem de pessoas em cenários de alta aglomeração (como plataformas de embarque e áreas públicas).

---

## 🛠️ Modificações e Padrões Implementados

- **Isolamento de Entradas**: Os arquivos de vídeo de entrada e os pesos dos modelos foram movidos para dentro de `/app/input/videos/` e `/app/weights/`, respectivamente.
- **Configurações Locais**: Os arquivos `data_day.yaml` e `data_night.yaml` foram atualizados para buscar os caminhos internos (`input/videos/...` e `weights/best.pt`).
- **Resolução de Caminhos Absolutos**: O módulo de configuração resolve todos os caminhos de forma absoluta em tempo de execução com base no diretório em que o arquivo `.yaml` correspondente se encontra.
- **Requerimentos Autônomos**: Um arquivo `requirements.txt` próprio foi adicionado à pasta `/app` com todas as dependências isoladas.
