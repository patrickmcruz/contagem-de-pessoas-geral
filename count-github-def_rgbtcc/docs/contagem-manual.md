Para resolver essa tarefa de anotação de dados (conhecida como *Point Annotation* ou anotação de *Keypoints*), você tem duas abordagens principais: criar um script rápido e customizado em Python ou usar uma ferramenta de anotação profissional já pronta.

Como seu objetivo final é treinar ou avaliar modelos de visão computacional (como estimativa de densidade de multidões), ter um arquivo limpo exportado em CSV ou JSON com as coordenadas `(X, Y)` é fundamental.

Aqui estão as melhores opções:

## Opção 1: Script Rápido em Python (OpenCV)

Se você quer apenas abrir a imagem, clicar e gerar um arquivo CSV instantaneamente no seu computador, o OpenCV resolve isso com poucas linhas de código.

1. Instale as dependências: `pip install opencv-python pandas`
2. Salve o código abaixo como `anotar.py` e execute:

```python
import cv2
import pandas as pd

# Lista para armazenar as coordenadas
coordenadas = []

def capturar_clique(event, x, y, flags, param):
    # Registra o ponto ao clicar com o botão esquerdo do mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        coordenadas.append({"x": x, "y": y})
        # Desenha um círculo vermelho no local clicado para feedback visual
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Anotacao de Multidao", img)
        print(f"Cabeça registrada em: X={x}, Y={y}")

# Carregue sua imagem aqui
caminho_imagem = "sua_imagem.jpg" 
img = cv2.imread(caminho_imagem)

if img is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    cv2.imshow("Anotacao de Multidao", img)
    cv2.setMouseCallback("Anotacao de Multidao", capturar_clique)
    
    print("Clique nas cabeças. Pressione 'ESC' ou 'q' para salvar e sair.")
    
    # Aguarda o usuário pressionar 'q' ou 'ESC'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): 
            break

    cv2.destroyAllWindows()

    # Exporta os dados para CSV
    if coordenadas:
        df = pd.DataFrame(coordenadas)
        df.to_csv("coordenadas_cabecas.csv", index=False)
        print(f"Salvo com sucesso! Total de pessoas marcadas: {len(coordenadas)}")

```

---

## Opção Pronta do Projeto: `scripts/anotar_pontos.py`

Para evitar problemas de falta de desfazer (*Undo*) ou anotações perdidas, criamos um script profissional completo localizado em:
`scripts/anotar_pontos.py`

### Como executar:
```bash
# Executa usando o ambiente virtual do projeto
notebooks/.venv/bin/python scripts/anotar_pontos.py
```

### Principais recursos:
- **Botão Esquerdo:** Adiciona ponto (cabeça do pedestre).
- **Botão Direito ou teclas `Z` / `U`:** Desfaz o último ponto anotado (*Undo*).
- **Tecla `C`:** Limpa todas as anotações.
- **Teclas `S`, `Q` ou `ESC`:** Salva e fecha.
- **HUD Integrado:** Mostra total de pessoas e posição do cursor em tempo real.
- **Artefatos Gerados (em `notebooks/DEF-rgbtcc/output/ground_truth/`):**
  1. `pontos_ground_truth.csv`: Coordenadas `(id, x, y)`.
  2. `pontos_ground_truth.json`: Metadados completos com dimensões e contagem total.
  3. `rgb_anotada_ground_truth.jpg`: Imagem com os pontos desenhados para apresentação e auditoria.

---

> **Dica para Visão Computacional:** Na literatura de contagem de multidões (*Crowd Counting*), esses pontos `(X, Y)` não são usados diretamente como *labels* para a rede neural. O processo padrão é passar esses pontos por um **Filtro Gaussiano** (Gaussian Kernel) para converter o fundo preto com os pontos em um **Mapa de Densidade** (Density Map) contínuo. A rede neural (como CSRNet ou MCNN) aprende a regredir esse mapa térmico em vez de prever coordenadas exatas.