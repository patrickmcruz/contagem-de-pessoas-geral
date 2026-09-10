#!/usr/bin/env python3
"""
Anotador Manual de Pontos (Ground Truth) para Contagem de Pessoas
===============================================================
Permite marcar interativamente a posição das cabeças/pedestres em imagens
RGB pré-processadas do pipeline RGBTCC, gerando os artefatos de Ground Truth
necessários para cálculo de métricas (MAE, MSE) e apresentações.

Atalhos:
- Botão Esquerdo: Marcar ponto (cabeça)
- Botão Direito ou 'Z' / 'U': Desfazer último ponto
- 'C': Limpar todas as anotações
- 'S' ou 'Q' ou ESC: Salvar anotações e sair
"""

import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import pandas as pd

# Caminhos padrão do projeto
DEFAULT_IMAGE = Path("notebooks/DEF-rgbtcc/output/01_pre_transformacao/rgb_preprocessed.jpg")
DEFAULT_OUTPUT_DIR = Path("notebooks/DEF-rgbtcc/output/ground_truth")

coordenadas = []
img_base = None
img_display = None
mouse_pos = (0, 0)
raio_ponto = 4


def atualizar_canvas():
    """Redesenha todos os pontos sobre a cópia da imagem base e projeta o HUD."""
    global img_display
    img_display = img_base.copy()
    h, w = img_base.shape[:2]

    # 1. Desenha os pontos anotados (círculo com borda para contraste em qualquer iluminação)
    for i, pt in enumerate(coordenadas):
        px, py = pt["x"], pt["y"]
        cv2.circle(img_display, (px, py), raio_ponto, (0, 0, 255), -1)      # Centro Vermelho
        cv2.circle(img_display, (px, py), raio_ponto + 2, (0, 255, 255), 1)  # Borda Amarela
        
        # Opcional: desenha o número se houver poucos pontos (< 100)
        if len(coordenadas) <= 60:
            cv2.putText(
                img_display,
                str(i + 1),
                (px + 6, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # 2. Barra de HUD superior (faixa escura semi-transparente)
    hud_h = 42
    overlay = img_display.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, img_display, 0.15, 0, img_display)
    cv2.line(img_display, (0, hud_h), (w, hud_h), (56, 189, 248), 2)

    # Textos do HUD
    texto_total = f"Pessoas Marcadas: {len(coordenadas)}"
    texto_coords = f"Cursor: ({mouse_pos[0]}, {mouse_pos[1]})"
    texto_ajuda = "[Esq]: Marcar | [Dir / Z]: Desfazer | [C]: Limpar | [S/Q/ESC]: Salvar e Sair"

    cv2.putText(
        img_display,
        texto_total,
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (74, 222, 128),  # Verde claro
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_display,
        texto_coords,
        (280, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (203, 213, 225),  # Cinza claro
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_display,
        texto_ajuda,
        (w - 680, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (147, 197, 253),  # Azul claro
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("Anotacao de Ground Truth (RGBTCC)", img_display)


def callback_mouse(event, x, y, flags, param):
    """Manipula eventos do cursor e cliques do mouse."""
    global coordenadas, mouse_pos
    mouse_pos = (x, y)

    # Movimento do mouse: atualiza posição no HUD
    if event == cv2.EVENT_MOUSEMOVE:
        atualizar_canvas()

    # Botão esquerdo: adiciona um novo ponto
    elif event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= y < img_base.shape[0] and 0 <= x < img_base.shape[1]:
            novo_id = len(coordenadas) + 1
            coordenadas.append({"id": novo_id, "x": int(x), "y": int(y)})
            print(f"[+] Ponto #{novo_id} anotado em: (x={x}, y={y}) | Total: {len(coordenadas)}")
            atualizar_canvas()

    # Botão direito: desfaz o último ponto inserido
    elif event == cv2.EVENT_RBUTTONDOWN:
        if coordenadas:
            removido = coordenadas.pop()
            print(f"[-] Ponto #{removido['id']} desfeito em: (x={removido['x']}, y={removido['y']}) | Restantes: {len(coordenadas)}")
            atualizar_canvas()


def carregar_anotacoes_existentes(caminho_csv: Path):
    """Carrega anotações prévias de um CSV para permitir continuação."""
    global coordenadas
    if caminho_csv.exists():
        try:
            df = pd.read_csv(caminho_csv)
            if "x" in df.columns and "y" in df.columns:
                coordenadas = [
                    {"id": i + 1, "x": int(row["x"]), "y": int(row["y"])}
                    for i, row in df.iterrows()
                ]
                print(f"[*] {len(coordenadas)} anotações carregadas previamente de: {caminho_csv}")
        except Exception as e:
            print(f"[!] Falha ao carregar anotações existentes: {e}")


def main():
    global img_base, raio_ponto
    parser = argparse.ArgumentParser(
        description="Anotador Interativo de Pontos de Multidão (Ground Truth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--imagem",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Caminho da imagem RGB (pós-transformação)",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório de saída dos artefatos de Ground Truth",
    )
    parser.add_argument(
        "--carregar",
        type=Path,
        default=None,
        help="Caminho de um CSV existente para continuar a anotação",
    )
    parser.add_argument(
        "--raio",
        type=int,
        default=4,
        help="Raio em pixels do marcador do ponto",
    )
    args = parser.parse_args()
    raio_ponto = args.raio

    # Fallback automático caso o caminho padrão de imagem não exista
    caminho_img = args.imagem
    if not caminho_img.exists():
        fallback_liu = Path("notebooks/liuzywen-RGBTCC/output/01_pre_transformacao/rgb_preprocessed.jpg")
        if fallback_liu.exists():
            print(f"[!] Imagem {caminho_img} não encontrada. Usando fallback: {fallback_liu}")
            caminho_img = fallback_liu
        else:
            print(f"[ERRO] Imagem não encontrada: {caminho_img}")
            print("Execute antes o notebook 01 de pré-transformação ou passe --imagem /caminho/imagem.jpg")
            sys.exit(1)

    print(f"[*] Carregando imagem: {caminho_img}")
    img_base = cv2.imread(str(caminho_img))
    if img_base is None:
        print(f"[ERRO] Não foi possível ler a imagem com OpenCV: {caminho_img}")
        sys.exit(1)

    # Cria diretório de saída
    args.saida.mkdir(parents=True, exist_ok=True)

    # Carrega arquivo existente se indicado
    p_csv_default = args.saida / "pontos_ground_truth.csv"
    if args.carregar:
        carregar_anotacoes_existentes(args.carregar)

    window_name = "Anotacao de Ground Truth (RGBTCC)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1280, 800)
    cv2.setMouseCallback(window_name, callback_mouse)

    atualizar_canvas()

    print("\n" + "=" * 70)
    print("       ANOTADOR INTERATIVO INICIADO COM SUCESSO")
    print("=" * 70)
    print("  • BOTÃO ESQUERDO:    Marcar ponto (cabeça)")
    print("  • BOTÃO DIREITO / Z: Desfazer o último ponto")
    print("  • TECLA 'C':         Limpar tudo")
    print("  • TECLA 'S' / 'Q':   Salvar e sair")
    print("=" * 70 + "\n")

    # Loop principal de eventos de teclado
    while True:
        key = cv2.waitKey(20) & 0xFF

        # Sair e Salvar (ESC, 'q', 's')
        if key in [27, ord("q"), ord("s"), ord("Q"), ord("S")]:
            break

        # Desfazer ('z' ou 'u')
        elif key in [ord("z"), ord("u"), ord("Z"), ord("U")]:
            if coordenadas:
                removido = coordenadas.pop()
                print(f"[-] Ponto #{removido['id']} desfeito em: (x={removido['x']}, y={removido['y']}) | Restantes: {len(coordenadas)}")
                atualizar_canvas()

        # Limpar ('c')
        elif key in [ord("c"), ord("C")]:
            if coordenadas:
                print("[!] Limpando todas as marcações.")
                coordenadas.clear()
                atualizar_canvas()

    cv2.destroyAllWindows()

    # Exportação dos arquivos finais
    p_csv = args.saida / "pontos_ground_truth.csv"
    p_json = args.saida / "pontos_ground_truth.json"
    p_img = args.saida / "rgb_anotada_ground_truth.jpg"

    # 1. Salvar CSV
    df = pd.DataFrame(coordenadas)
    df.to_csv(p_csv, index=False)

    # 2. Salvar JSON
    telemetria_gt = {
        "data_criacao": time.strftime("%Y-%m-%d %H:%M:%S"),
        "imagem_origem": str(caminho_img),
        "resolucao": {
            "largura": int(img_base.shape[1]),
            "altura": int(img_base.shape[0]),
            "canais": int(img_base.shape[2]) if len(img_base.shape) > 2 else 1,
        },
        "total_pessoas_anotadas": len(coordenadas),
        "pontos": coordenadas,
    }
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(telemetria_gt, f, indent=2, ensure_ascii=False)

    # 3. Salvar Imagem Anotada
    cv2.imwrite(str(p_img), img_display)

    print("\n" + "=" * 70)
    print("       ENTREGÁVEIS DE GROUND TRUTH GERADOS COM SUCESSO")
    print("=" * 70)
    print(f"  [✓] Total de Pessoas Anotadas: {len(coordenadas)}")
    print(f"  [✓] Tabela CSV de Coordenadas: {p_csv}")
    print(f"  [✓] Metadados e Pontos JSON:   {p_json}")
    print(f"  [✓] Imagem com Auditoria GT:   {p_img}")
    print("=" * 70)


if __name__ == "__main__":
    main()
