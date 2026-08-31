# Sistema de Contagem de Pessoas (Live)

Este diretório contém a implementação profissional e modular da contagem de pessoas com Re-Identificação usando a **webcam** como entrada.

## 🏗️ Arquitetura Modular
O projeto utiliza uma arquitetura baseada em componentes, separada em:
- **Core**: Interfaces base para extensibilidade.
- **Detectors**: Implementação YOLOv11 para detecção de objetos.
- **ReID**: Extração de assinaturas visuais (MobileNetV3).
- **Tracking**: Lógica de estabilidade e fusão de identidades.
- **UI**: Processamento de vídeo e renderização de overlays.

## 🚀 Como Executar
Certifique-se de que possui as dependências instaladas (`pip install -r requirements.txt`).

### Execução Principal
Para iniciar a detecção em tempo real via webcam:
```bash
python src/main.py
```

### Ambiente de Pesquisa (Notebooks)
Para prototipagem e testes rápidos, veja a pasta `notebooks/`.

### Execução de Testes
Para validar a lógica de rastreamento:
```bash
python -m pytest tests/unit/test_tracker.py
```

## 🧠 Lógica de Contagem Estável
1. **Filtro de Estabilidade**: Apenas pessoas vistas por 20 frames consecutivos são contadas no total geral.
2. **Re-Identificação (Re-ID)**: Utiliza similaridade de cosseno (threshold 0.72) para reconhecer pessoas que saem e voltam.
3. **ID Merging**: Se um novo ID for muito parecido com um já existente (merge threshold 0.82), os IDs são fundidos retroativamente.
