# Sistema de Contagem de Pessoas (Versão Profissional)

Este diretório contém a implementação profissional e modular da contagem de pessoas com Re-Identificação.

## 🏗️ Arquitetura Modular
O projeto utiliza uma arquitetura baseada em componentes, separando as responsabilidades em:
- **Core**: Interfaces base para extensibilidade.
- **Detectors**: Implementação YOLOv11n para detecção rápida.
- **ReID**: Extração de assinaturas visuais via MobileNetV3.
- **Tracking**: Lógica de estabilidade e fusão de identidades.
- **UI**: Processamento de vídeo e renderização de overlays.

## 🚀 Como Executar
Garanta que possui as dependências instaladas (`ultralytics`, `torch`, `torchvision`, `pytest`).

### Execução Principal
Para iniciar a detecção em tempo real via webcam com janela nativa OpenCV:
```bash
python src/main.py
```

### Execução de Testes
Para garantir que a lógica de contagem está funcionando corretamente:
```bash
python -m pytest tests/unit/test_tracker.py
```

## 🧠 Lógica de Contagem Estável
1. **Filtro de Estabilidade**: Apenas pessoas vistas por 20 frames consecutivos são contadas no total geral.
2. **Re-Identificação (Re-ID)**: Utiliza similaridade de cosseno (threshold 0.72) para reconhecer pessoas que saem e voltam.
3. **ID Merging**: Se um novo ID for criado mas for muito parecido com um já existente (merge threshold 0.82), os IDs são fundidos retroativamente para manter o contador preciso.
