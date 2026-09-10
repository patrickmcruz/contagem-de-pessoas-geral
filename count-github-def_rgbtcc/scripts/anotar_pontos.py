#!/usr/bin/env python3
"""
Anotador Manual de Pontos (Ground Truth) para Contagem de Pessoas
===============================================================
Permite marcar interativamente a posição das cabeças/pedestres em imagens
RGB pré-processadas do pipeline RGBTCC, gerando os artefatos de Ground Truth
necessários para cálculo de métricas (MAE, MSE) e apresentações.

Atalhos & Controles:
- Botão Esquerdo (na imagem): Marcar ponto (cabeça)
- Botão Esquerdo (no topo): Clicar no botão interativo "[ ↩ Desfazer ]" ou "[ 💾 Salvar ]"
- Botão Direito (em qualquer lugar): Desfazer último ponto
- Teclas Ctrl+Z, Z, U, Backspace ou Seta Esquerda: Desfazer último ponto
- Tecla C: Limpar todas as anotações
- Teclas S, Q ou ESC: Salvar anotações e sair
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
raio_ponto = 4
deve_encerrar = False

# Coordenadas dos botões no HUD superior
HUD_HEIGHT = 48
BTN_UNDO = (0, 0, 0, 0)
BTN_SAVE = (0, 0, 0, 0)


def atualizar_canvas():
    """Redesenha todos os pontos sobre a imagem base e projeta o HUD interativo."""
    global img_display, BTN_UNDO, BTN_SAVE
    img_display = img_base.copy()
    h, w = img_base.shape[:2]

    # 1. Desenha os pontos anotados (círculo com borda para contraste)
    for i, pt in enumerate(coordenadas):
        px, py = pt["x"], pt["y"]
        cv2.circle(img_display, (px, py), raio_ponto, (0, 0, 255), -1)       # Vermelho
        cv2.circle(img_display, (px, py), raio_ponto + 2, (0, 255, 255), 1)  # Borda Amarela
        
        # Numeração dos primeiros pontos para auditoria
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

    # 2. Faixa do HUD superior
    overlay = img_display.copy()
    cv2.rectangle(overlay, (0, 0), (w, HUD_HEIGHT), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.90, img_display, 0.10, 0, img_display)
    cv2.line(img_display, (0, HUD_HEIGHT), (w, HUD_HEIGHT), (56, 189, 248), 2)

    # 3. Informações à esquerda: Contador de pessoas
    cv2.putText(
        img_display,
        f"Pessoas: {len(coordenadas)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (74, 222, 128),  # Verde
        2,
        cv2.LINE_AA,
    )

    # 4. Instruções no centro
    cv2.putText(
        img_display,
        "Botao Esq: Marcar | Botao Dir / Ctrl+Z / <- : Desfazer",
        (220, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (203, 213, 225),  # Cinza claro
        1,
        cv2.LINE_AA,
    )

    # 5. Botões clicáveis à direita
    # Botão Desfazer (Laranja/Âmbar)
    u_x1, u_y1, u_x2, u_y2 = w - 360, 8, w - 190, 40
    BTN_UNDO = (u_x1, u_y1, u_x2, u_y2)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (30, 110, 230), -1)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "<- Desfazer",
        (u_x1 + 18, u_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Botão Salvar (Verde esmeralda)
    s_x1, s_y1, s_x2, s_y2 = w - 175, 8, w - 15, 40
    BTN_SAVE = (s_x1, s_y1, s_x2, s_y2)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (40, 150, 60), -1)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "Salvar & Sair",
        (s_x1 + 16, s_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Anotacao de Ground Truth (RGBTCC)", img_display)


def desfazer_ultimo_ponto():
    """Remove o último ponto adicionado e atualiza a interface."""
    global coordenadas
    if coordenadas:
        removido = coordenadas.pop()
        print(f"[-] Ponto #{removido['id']} desfeito em: (x={removido['x']}, y={removido['y']}) | Restantes: {len(coordenadas)}")
        atualizar_canvas()
    else:
        print("[!] Nenhum ponto registrado para desfazer.")


def callback_mouse(event, x, y, flags, param):
    """Manipula cliques do mouse e botões na tela."""
    global coordenadas, deve_encerrar

    # Botão esquerdo
    if event == cv2.EVENT_LBUTTONDOWN:
        # Verifica se clicou na área do HUD (botões da interface)
        if y <= HUD_HEIGHT:
            u_x1, u_y1, u_x2, u_y2 = BTN_UNDO
            s_x1, s_y1, s_x2, s_y2 = BTN_SAVE

            # Clicou no botão [ Desfazer ]
            if u_x1 <= x <= u_x2 and u_y1 <= y <= u_y2:
                desfazer_ultimo_ponto()
                return

            # Clicou no botão [ Salvar & Sair ]
            elif s_x1 <= x <= s_x2 and s_y1 <= y <= s_y2:
                print("[*] Botão 'Salvar & Sair' acionado pelo mouse.")
                deve_encerrar = True
                return
            return

        # Clique dentro da imagem: adiciona um novo ponto
        if 0 <= y < img_base.shape[0] and 0 <= x < img_base.shape[1]:
            novo_id = len(coordenadas) + 1
            coordenadas.append({"id": novo_id, "x": int(x), "y": int(y)})
            print(f"[+] Ponto #{novo_id} anotado em: (x={x}, y={y}) | Total: {len(coordenadas)}")
            atualizar_canvas()

    # Botão direito: desfaz em qualquer lugar da tela
    elif event == cv2.EVENT_RBUTTONDOWN:
        desfazer_ultimo_ponto()


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
    global img_base, raio_ponto, deve_encerrar
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
    # WINDOW_NORMAL com flag GUI_NORMAL remove menus irrelevantes do Qt mantendo janela responsiva
    flags_win = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 16)
    cv2.namedWindow(window_name, flags_win)
    cv2.resizeWindow(window_name, 1280, 800)
    cv2.setMouseCallback(window_name, callback_mouse)

    atualizar_canvas()

    print("\n" + "=" * 70)
    print("       ANOTADOR INTERATIVO INICIADO COM SUCESSO")
    print("=" * 70)
    print("  • BOTÃO ESQUERDO:    Marcar ponto (ou clicar em [<- Desfazer] no HUD)")
    print("  • BOTÃO DIREITO:     Desfazer último ponto em qualquer lugar")
    print("  • TECLADO DESFAZER:  Ctrl+Z, Z, U, Backspace ou Seta Esquerda (<-)")
    print("  • TECLA 'C':         Limpar tudo")
    print("  • TECLA 'S' / 'Q':   Salvar e sair (ou clicar em [Salvar & Sair] no HUD)")
    print("=" * 70 + "\n")

    # Loop principal de eventos de teclado com waitKeyEx para capturar teclas especiais
    while not deve_encerrar:
        raw_key = cv2.waitKeyEx(30)
        if raw_key == -1:
            continue

        key = raw_key & 0xFF

        # Sair e Salvar:
        # ESC (27), 's'/'S' (115/83), 'q'/'Q' (113/81 se não for seta)
        if raw_key in [27, ord("s"), ord("S")] or (key in [ord("q"), ord("Q")] and raw_key not in [65361, 81]):
            break

        # Desfazer (Undo):
        # 1. Ctrl+Z (ASCII 26)
        # 2. 'z' (122), 'Z' (90), 'u' (117), 'U' (85)
        # 3. Backspace (8) ou Delete (127, 255, 65535)
        # 4. Seta Esquerda (Linux X11/Qt: 65361 ou 81 ou 2424832)
        elif raw_key in [26, 65361, 8, 127, 65535, 2424832] or key in [26, ord("z"), ord("Z"), ord("u"), ord("U"), 8, 127]:
            desfazer_ultimo_ponto()

        # Limpar ('c' / 'C')
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
