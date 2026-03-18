# Detecção de Objetos com YOLOv11

Este projeto utiliza o estado da arte em visão computacional para realizar a detecção de objetos em tempo real através da webcam, otimizado para rodar em GPUs NVIDIA.

## 🚀 Tecnologias Utilizadas

- **Python 3.10+**
- **YOLOv11** (via biblioteca `ultralytics`)
- **PyTorch** (com suporte a CUDA)
- **OpenCV** (para captura e exibição de vídeo)

## 💻 Requisitos de Hardware (Ref. do Usuário)

- **GPU:** NVIDIA RTX 1660 Super (8GB)
- **CPU:** AMD Ryzen 5 3600
- **RAM:** 16GB

## 🛠️ Instalação e Configuração

### 1. Clonar o repositório (ou baixar os arquivos)
```bash
git clone https://github.com/patrickmcruz/contagem-de-pessoas.git
cd contagem-de-pessoas
```

### 2. Instalar dependências básicas
```bash
pip install ultralytics opencv-python torch torchvision
```

### 3. Ativar Suporte a GPU (NVIDIA)
Para garantir que o PyTorch utilize sua RTX 1660 Super, instale a versão com suporte a CUDA:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📖 Como Usar

1. Abra o arquivo [object_detection.ipynb](object_detection.ipynb) no VS Code ou Jupyter Notebook.
2. Execute as células em ordem.
3. A janela da câmera abrirá focando apenas na classe `person`.
4. Pressione a tecla **'q'** para fechar a janela e encerrar o script.

## ⚠️ Troubleshooting (Problemas Comuns)

### Erro: `ValueError: numpy.dtype size changed`
Este erro ocorre devido à incompatibilidade com o NumPy 2.0. Para corrigir:
1. Execute: `pip install "numpy<2"`
2. **Reinicie o Kernel** do seu notebook/editor.

---
Desenvolvido como um exemplo de Visão Computacional de alta performance.
