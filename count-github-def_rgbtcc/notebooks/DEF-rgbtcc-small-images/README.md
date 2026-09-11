# Pipeline DEF-rgbtcc para Imagens Menores e Poucas Pessoas (Small Images)

Este diretório contém a suíte completa de notebooks interativos dedicados ao estudo de **recortes de pequenas dimensões com baixa densidade de pedestres**, utilizando a arquitetura **DEF-rgbtcc**.

## Estrutura dos Notebooks

1. **[`01_pre_transformacao_recorte.ipynb`](01_pre_transformacao_recorte.ipynb):**
   - Seleção de amostras curadas (0, 5, 9 e 19 pessoas).
   - Equalização local CLAHE na imagem térmica para realce de silhuetas.
   - Padronização de dimensões para múltiplos exatos de 32 pixels.
   - Auditoria visual por blend 50% RGB / 50% Térmica.
   - Exportação do contrato de dados em `output/01_pre_transformacao/`.

2. **[`02_contagem_pessoas_recorte.ipynb`](02_contagem_pessoas_recorte.ipynb):**
   - Ingestão do contrato padronizado.
   - Inferência neural em tempo real (< 20 ms).
   - Dupla estratégia de contagem: **Integral Contínua** vs **Picos Locais (Filtragem de Ruído)**.
   - Confronto imediato com Ground Truth humano.
   - Painel de auditoria executivo em 5 colunas salvo em `output/02_contagem/`.
   - **Métricas Avançadas de Validação:** Cálculo e gráfico de resíduos de **MSE** (fidelidade do mapa 2D) e **NAE** (erro absoluto normalizado).
   - Apresentação executiva em texto formatado e card visual HTML.


3. **[`03_estudo_densidade_e_metricas_poucas_pessoas.ipynb`](03_estudo_densidade_e_metricas_poucas_pessoas.ipynb):**
   - Processamento em lote de todas as amostras curadas.
   - Gráfico de correlação e análise de desvio em baixa densidade.
   - Investigação de falso-positivo no controle negativo (0 pessoas).
   - Tabela de métricas consolidadas ($MAE$, $RMSE$) e recomendações de projeto.

## Diretriz Didática

Cada cabeçalho de célula foi elaborado seguindo a diretriz pedagógica:
- **O que o código faz:** Resumo direto da operação.
- **Por que esta lógica foi escolhida:** Explicação técnica acessível sobre a decisão de engenharia.
- **Efeito prático no resultado:** O que esperar da saída visual ou numérica.
