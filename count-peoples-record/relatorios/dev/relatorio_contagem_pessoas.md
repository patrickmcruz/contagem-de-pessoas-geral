# Relatório de Pesquisa: Sistema de Contagem de Público

Este documento detalha a metodologia técnica e as tecnologias aplicadas no desenvolvimento do sistema de contagem de pessoas, consolidando os resultados de pesquisa para o projeto de monitoramento de público.

## 1. Introdução

O objetivo central desta pesquisa foi criar uma solução robusta para a contagem de indivíduos em vídeos gravados, capaz de lidar com desafios como oclusões temporárias, mudanças de perspectiva e a necessidade de evitar a contagem duplicada da mesma pessoa.

## 2. Metodologia Aplicada

A metodologia segue um pipeline de Visão Computacional de última geração, estruturado em cinco etapas principais:

### A. Análise dos Dados de Entrada
O sistema foi validado utilizando uma base de vídeos com características de alta resolução, garantindo a precisão necessária para a detecção de pequenos detalhes:
- **Resoluções**: Predominantemente 4K (3840x2160 pixels) e Full HD (1920x1080 pixels).
- **Taxa de Quadros (FPS)**: Variável entre 26 e 58 FPS, permitindo testar a robustez do rastreador em diferentes dinâmicas de movimento.
- **Duração**: Amostras de 7 a 23 segundos, focadas em fluxos intensos de passagem.

### B. Detecção de Objetos (YOLOv11)
A primeira camada do sistema utiliza o modelo **YOLOv11** (You Only Look Once). Este modelo processa cada quadro do vídeo em busca da classe "pessoa". A escolha do YOLOv11 garante alta precisão e velocidade.

### C. Extração de Características (Re-Identification)
Para garantir que um indivíduo não seja contado duas vezes, aplicamos a rede **MobileNetV3 Large** para extrair "vetores de identidade" (embeddings) de cada pessoa detectada, funcionando como uma assinatura visual única.

### D. Rastreador Estável (StableTracker)
O algoritmo utiliza **Similaridade de Cosseno** para comparar embeddings entre quadros. Implementamos uma lógica de **Merge (Fusão)** para manter a continuidade do ID mesmo após perdas momentâneas de detecção.

### E. Estabilização e Contagem Confirmada
O contador global apenas incrementa após uma pessoa ser vista por um número mínimo de quadros consecutivos, filtrando ruídos e falsos positivos.

### F. Processamento de Saída
O vídeo final preserva o codec `mp4v`, dimensões e FPS originais, adicionando anotações de Bounding Boxes, IDs e uma barra de status com a contagem acumulada.

## 3. Tecnologias Escolhidas

| Nível | Tecnologia | Papel no Projeto |
| :--- | :--- | :--- |
| **Linguagem Base** | **Python 3.12** | Base do ecossistema. |
| **Processamento** | **OpenCV** | Manipulação de vídeo e anotações. |
| **Deep Learning** | **MobileNetV3** | Extração de Re-ID. |
| **IA SOTA** | **YOLOv11** | Detecção espacial. |

## 4. Otimização de Performance

O sistema foi otimizado para hardware de alto desempenho:
- **Aceleração por GPU**: CUDA Habilitado.
- **Precisão Mista (FP16)**: Máxima velocidade na **RTX 4090**.

## 5. Resultados Obtidos


### Amostra 01
- **Arquivo**: `exemplo_3_pessoa_input.mp4` (14 segundos, 45.80 FPS). Resolução 2160 x 3840.
- **Camera**: vídeo vertical capturado diretamente do Poco X7 Pro, com resolução 4K e 60fps.
- **Detalhes do conteúdo**: Pessoa sentada na cadeira do escritório, sendo identificada e contabilizada pelo sistema. Camera se move e mostra rosto do camera-man e sistema registra corretamente. Camera se move e mostra pés do camera-man, que é identificado como uma nova pessoa e contabilizado. Uma nova pessoa sentada na cadeira é filmada e sistema contabiliza novamente. Camera sai rapidamente e retorno para a mesma pessoa, e sistema não registra novo indivíduo. Camera filma a mão do camera man a qual é contabilizada.
- **Eficiência GPU**: 72.60 segundos
- **Eficiência CPU**: 40 minutos.
- **Precisão**: 5 pessoas únicas detectadas de 3 existentes.

Os testes de validação apresentaram os seguintes indicadores:
### Amostra 02
- **Arquivo**: `exemplo_4_pessoa_input.mp4` (22 segundos, 26.26 FPS). Resolução 1920 x 1280.
- **Camera**: vídeo horizontal capturado diretamente da webcam logitech c920s, com resolução 1080p e 30fps.
- **Detalhes do conteúdo**: Pessoa em pé, sendo identificada e contabilizada pelo sistema. Pessoa sai rapidamente do quadro e entra novamente: sistema consegue identificar e contabilizar novamente. Pessoa sai do quadro por 5 segundos e entra novamente: sistema consegue identificar e contabilizar novamente. Pessoa sai rapidamente do quadro põe óculos e entra novamente: sistema identifica outro indivíduo. Pessoa de óculos sai do quadro e entra novamente: sistema consegue identificar e contabilizar novamente. Ao final duas pessoas entram no quadro e o sistema consegue identificar e contabilizar.
- **Eficiência GPU**: 58.38 segundos
- **Eficiência CPU**: 20 minutos.
- **Precisão**: 4 pessoas únicas detectadas de 3 existentes.

## 6. Conclusão da Pesquisa

- A metodologia de "Rastreamento Estável" permite uma contagem acumulada confiável, boa para a análise de fluxo de público e planejamento de espaços físicos. Porém exigem grande poder computacional para pós-processamento.

## 7. Trabalhos Futuros
- Refinamento de hiper-parâmetros para otimização de performance do modelo, objetivando testes com resultados diversos.
- Processamento com vídeos em oclusão maior e baixa luminosidade.
- Processamento com vídeos obtidos de câmeras sensíveis a ambientes noturnos.
- Processamento de imagem em servidor de CPU do Hipervisor com múltiplos processadores mais poderosos
- Processamento com vídeos de eventos reais.
- Fine-tuning (treinamento) de modelos usando imagens de locais específicos.
- Exploração de modelos de inferência para contagem de multidões como P2PNet, CSRNet, etc.
