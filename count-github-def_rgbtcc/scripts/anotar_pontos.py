#!/usr/bin/env python3
"""
Anotador Manual de Pontos (Ground Truth) para Contagem de Pessoas
===============================================================
Exibe a imagem em FULLSCREEN com ajuste perfeito ('fit' proporcional) à tela,
permitindo marcar interativamente as pessoas na imagem original de alta resolução
(8000x6000 px DJI_0789_W.JPG), registrando as coordenadas reais do sensor.

Recursos:
- Abertura em Fullscreen com 'Fit' automático à resolução do monitor.
- Imagem centralizada sem estourar a tela e sem distorcer proporções.
- HUD superior limpo e nítido com botões interativos clicáveis.
- Mapeamento bidirecional exato entre a tela e a imagem original 8000x6000.
- Checkpoint / Salvar Progresso: Salva o estado atual sem fechar a janela.
- Auto-Continuação: Ao abrir a mesma imagem, carrega os pontos anteriores.
- Finalizar: Encerra a contagem e gera os entregáveis de Ground Truth.

Atalhos & Controles:
- Botão Esquerdo (na imagem): Marcar ponto (cabeça)
- Botão [ <- Desfazer ] (ou Ctrl+Z / Seta <- / Botão Direito): Desfazer último ponto
- Botão [ 💾 Salvar ] (ou Ctrl+S / S): Salvar progresso atual (checkpoint)
- Botão [ ✓ Finalizar ] (ou F / ESC): Finalizar a contagem e gerar entregáveis
- Tecla F11: Alternar tela cheia / janela
- Tecla C: Limpar todas as marcações
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Caminhos padrão do projeto (Imagem óptica original 8000x6000 px)
DEFAULT_IMAGE = Path("notebooks/DEF-rgbtcc/input/DJI_0789_W.JPG")
DEFAULT_OUTPUT_DIR = Path("notebooks/DEF-rgbtcc/output/ground_truth")

coordenadas = []
img_base = None             # Imagem original 8000x6000 em memória
canvas_base = None          # Canvas de fundo com imagem redimensionada ('fit')
img_display = None          # Frame final renderizado para cv2.imshow
raio_marcador_display = 1   # Raio padrão ultra-preciso na tela (1px)
is_fullscreen = True

# Variáveis de projeção geométrica ('fit')
scale_fit = 1.0
offset_x = 0
offset_y = 0
fit_w = 0
fit_h = 0
screen_w = 1920
screen_h = 1080
HUD_HEIGHT = 52

deve_encerrar = False
status_mensagem = "Botao Esq: Marcar | Botao Dir: Desfazer | [+/-]: Tamanho do Ponto | F11: Tela Cheia"
status_cor = (203, 213, 225)

# Diretório e imagem ativos
caminho_img_ativo = None
caminho_saida_ativo = None

# Botões interativos no HUD
BTN_UNDO = (0, 0, 0, 0)
BTN_SAVE = (0, 0, 0, 0)
BTN_FINISH = (0, 0, 0, 0)


def obter_resolucao_tela(default_w=1920, default_h=1080):
    """Detecta automaticamente a resolução do monitor no Linux (Wayland/X11)."""
    try:
        from pathlib import Path
        for p in sorted(Path("/sys/class/drm").glob("card*-*/modes")):
            lines = p.read_text().splitlines()
            if lines:
                parts = lines[0].strip().split("x")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
                    if w >= 800 and h >= 600:
                        return w, h
    except Exception:
        pass
    return default_w, default_h


def salvar_checkpoint(silencioso: bool = False):
    """Salva o progresso atual em disco (checkpoint) sem fechar a aplicação."""
    global status_mensagem, status_cor
    if caminho_saida_ativo is None or caminho_img_ativo is None:
        return

    caminho_saida_ativo.mkdir(parents=True, exist_ok=True)
    stem = caminho_img_ativo.stem
    p_csv_stem = caminho_saida_ativo / f"pontos_ground_truth_{stem}.csv"
    p_csv_default = caminho_saida_ativo / "pontos_ground_truth.csv"
    p_json_stem = caminho_saida_ativo / f"checkpoint_{stem}.json"
    p_json_default = caminho_saida_ativo / "checkpoint_anotacao.json"

    # Salva CSV
    df = pd.DataFrame(coordenadas)
    df.to_csv(p_csv_stem, index=False)
    df.to_csv(p_csv_default, index=False)

    # Salva JSON de Checkpoint
    checkpoint_data = {
        "tipo": "checkpoint_progresso",
        "data_checkpoint": time.strftime("%Y-%m-%d %H:%M:%S"),
        "imagem_origem": str(caminho_img_ativo),
        "arquivo_nome": caminho_img_ativo.name,
        "resolucao_original": {
            "largura": int(img_base.shape[1]),
            "altura": int(img_base.shape[0]),
        },
        "total_pessoas_anotadas": len(coordenadas),
        "pontos": coordenadas,
    }
    with open(p_json_stem, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    with open(p_json_default, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    hora_str = time.strftime("%H:%M:%S")
    status_mensagem = f"✓ Salvo as {hora_str} ({len(coordenadas)} pts)"
    status_cor = (74, 222, 128)  # Verde

    if not silencioso:
        print(f"[✓ CHECKPOINT] {len(coordenadas)} anotações salvas em: {p_csv_stem.name} ({hora_str})")
    atualizar_canvas()


def construir_canvas_fit():
    """Gera a base da tela com a imagem 8000x ajustada proporcionalmente ('fit')."""
    global canvas_base, scale_fit, offset_x, offset_y, fit_w, fit_h
    h_orig, w_orig = img_base.shape[:2]

    avail_w = screen_w
    avail_h = screen_h - HUD_HEIGHT

    # Escala proporcional garantindo 'fit' perfeito (aspect ratio 4:3)
    scale_fit = min(avail_w / w_orig, avail_h / h_orig)
    fit_w = int(round(w_orig * scale_fit))
    fit_h = int(round(h_orig * scale_fit))

    # Centralização
    offset_x = (avail_w - fit_w) // 2
    offset_y = HUD_HEIGHT + (avail_h - fit_h) // 2

    # Canvas com fundo dark slate moderno
    canvas_base = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas_base[:] = (18, 22, 30)

    # Redimensionamento suave da imagem original
    img_resized = cv2.resize(img_base, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
    canvas_base[offset_y:offset_y + fit_h, offset_x:offset_x + fit_w] = img_resized

    # Moldura sutil ao redor da imagem
    cv2.rectangle(
        canvas_base,
        (offset_x - 1, offset_y - 1),
        (offset_x + fit_w, offset_y + fit_h),
        (51, 65, 85),
        1,
    )


def atualizar_canvas():
    """Renderiza os pontos e o HUD sobre o canvas ajustado à tela."""
    global img_display, BTN_UNDO, BTN_SAVE, BTN_FINISH
    img_display = canvas_base.copy()
    w = screen_w

    # 1. Desenha os pontos anotados mapeados para a tela
    for i, pt in enumerate(coordenadas):
        disp_x = int(round(pt["x"] * scale_fit)) + offset_x
        disp_y = int(round(pt["y"] * scale_fit)) + offset_y

        if raio_marcador_display <= 1:
            # Ponto ultra sutil e preciso: ponto central vermelho puro
            cv2.circle(img_display, (disp_x, disp_y), 1, (0, 0, 255), -1)
        else:
            cv2.circle(img_display, (disp_x, disp_y), raio_marcador_display, (0, 0, 255), -1)      # Vermelho
            cv2.circle(img_display, (disp_x, disp_y), raio_marcador_display + 1, (0, 255, 255), 1) # Borda Amarela

    # 2. Barra de HUD superior
    cv2.rectangle(img_display, (0, 0), (w, HUD_HEIGHT), (15, 23, 42), -1)
    cv2.line(img_display, (0, HUD_HEIGHT), (w, HUD_HEIGHT), (56, 189, 248), 2)

    # 3. Informações à esquerda: Contador de pessoas
    cv2.putText(
        img_display,
        f"Pessoas: {len(coordenadas)}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (74, 222, 128),  # Verde
        2,
        cv2.LINE_AA,
    )

    # 4. Status e instruções no centro
    cv2.putText(
        img_display,
        status_mensagem,
        (210, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        status_cor,
        1,
        cv2.LINE_AA,
    )

    # 5. Botões interativos à direita
    # Botão 1: Desfazer (Laranja/Âmbar)
    u_x1, u_y1, u_x2, u_y2 = w - 460, 8, w - 330, 44
    BTN_UNDO = (u_x1, u_y1, u_x2, u_y2)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (30, 110, 230), -1)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "<- Desfazer",
        (u_x1 + 14, u_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Botão 2: Salvar Progresso (Azul Celeste)
    s_x1, s_y1, s_x2, s_y2 = w - 315, 8, w - 165, 44
    BTN_SAVE = (s_x1, s_y1, s_x2, s_y2)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (180, 105, 14), -1)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "Salvar (Ctrl+S)",
        (s_x1 + 10, s_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Botão 3: Finalizar (Verde Esmeralda)
    f_x1, f_y1, f_x2, f_y2 = w - 150, 8, w - 15, 44
    BTN_FINISH = (f_x1, f_y1, f_x2, f_y2)
    cv2.rectangle(img_display, (f_x1, f_y1), (f_x2, f_y2), (40, 150, 60), -1)
    cv2.rectangle(img_display, (f_x1, f_y1), (f_x2, f_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "V Finalizar",
        (f_x1 + 18, f_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Anotacao de Ground Truth (RGBTCC)", img_display)


def desfazer_ultimo_ponto():
    """Remove o último ponto adicionado e atualiza a interface."""
    global coordenadas, status_mensagem, status_cor
    if coordenadas:
        removido = coordenadas.pop()
        status_mensagem = f"Ponto #{removido['id']} desfeito ({len(coordenadas)} restantes)"
        status_cor = (147, 197, 253)
        print(f"[-] Ponto #{removido['id']} desfeito em: (x={removido['x']}, y={removido['y']}) | Restantes: {len(coordenadas)}")
        atualizar_canvas()
    else:
        status_mensagem = "Nenhum ponto para desfazer"
        status_cor = (248, 113, 113)
        print("[!] Nenhum ponto registrado para desfazer.")
        atualizar_canvas()


def callback_mouse(event, x, y, flags, param):
    """Manipula cliques do mouse na tela e mapeia para a resolução original 8000x6000."""
    global coordenadas, deve_encerrar, status_mensagem, status_cor

    # Botão esquerdo
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Clique na área do HUD superior
        if y <= HUD_HEIGHT:
            u_x1, u_y1, u_x2, u_y2 = BTN_UNDO
            s_x1, s_y1, s_x2, s_y2 = BTN_SAVE
            f_x1, f_y1, f_x2, f_y2 = BTN_FINISH

            # Botão [ Desfazer ]
            if u_x1 <= x <= u_x2 and u_y1 <= y <= u_y2:
                desfazer_ultimo_ponto()
                return

            # Botão [ Salvar ]
            elif s_x1 <= x <= s_x2 and s_y1 <= y <= s_y2:
                salvar_checkpoint()
                return

            # Botão [ Finalizar ]
            elif f_x1 <= x <= f_x2 and f_y1 <= y <= f_y2:
                print("[*] Botão 'Finalizar' acionado pelo usuário.")
                deve_encerrar = True
                return
            return

        # 2. Clique dentro dos limites da imagem exibida ('fit')
        if offset_x <= x < offset_x + fit_w and offset_y <= y < offset_y + fit_h:
            h_orig, w_orig = img_base.shape[:2]

            # Mapeamento para coordenadas originais do sensor (8000x6000)
            orig_x = int(round((x - offset_x) / scale_fit))
            orig_y = int(round((y - offset_y) / scale_fit))

            orig_x = max(0, min(w_orig - 1, orig_x))
            orig_y = max(0, min(h_orig - 1, orig_y))

            novo_id = len(coordenadas) + 1
            coordenadas.append({"id": novo_id, "x": orig_x, "y": orig_y})
            status_mensagem = f"Ponto #{novo_id} anotado em ({orig_x}, {orig_y}) | Total: {len(coordenadas)}"
            status_cor = (203, 213, 225)
            print(f"[+] Ponto #{novo_id} anotado: Tela=({x}, {y}) -> Original=({orig_x}, {orig_y}) | Total: {len(coordenadas)}")
            atualizar_canvas()

    # Botão direito: desfaz em qualquer lugar da tela
    elif event == cv2.EVENT_RBUTTONDOWN:
        desfazer_ultimo_ponto()


def carregar_anotacoes(caminho_arquivo: Path, img_shape=None) -> int:
    """Carrega anotações prévias de um CSV ou JSON validando compatibilidade."""
    global coordenadas
    if not caminho_arquivo.exists():
        return 0

    try:
        if caminho_arquivo.suffix.lower() == ".json":
            with open(caminho_arquivo, "r", encoding="utf-8") as f:
                dados = json.load(f)

            pontos = dados.get("pontos", [])
            coordenadas = [
                {"id": i + 1, "x": int(pt["x"]), "y": int(pt["y"])}
                for i, pt in enumerate(pontos)
            ]
        else:
            df = pd.read_csv(caminho_arquivo)
            if "x" in df.columns and "y" in df.columns:
                # Proteção contra carregar coordenadas de imagem 1280x na de 8000x
                if img_shape is not None and len(df) > 0:
                    max_x, max_y = df["x"].max(), df["y"].max()
                    if img_shape[1] > 4000 and max_x < 1500 and max_y < 1200:
                        print(f"[AVISO] As coordenadas em {caminho_arquivo.name} parecem ser de imagem reduzida (<1500px). Ignorado para a imagem 8000x.")
                        return 0

                coordenadas = [
                    {"id": i + 1, "x": int(row["x"]), "y": int(row["y"])}
                    for i, row in df.iterrows()
                ]
        return len(coordenadas)
    except Exception as e:
        print(f"[!] Falha ao ler anotações prévias de {caminho_arquivo}: {e}")
        return 0


def main():
    global img_base, raio_marcador_display, deve_encerrar, caminho_img_ativo, caminho_saida_ativo
    global screen_w, screen_h, is_fullscreen, status_mensagem, status_cor

    parser = argparse.ArgumentParser(
        description="Anotador Interativo em Fullscreen com 'Fit' para Imagens 8000x",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--imagem",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Caminho da imagem RGB (padrão: original 8000x DJI_0789_W.JPG)",
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
        help="Caminho específico de um CSV/JSON para carregar anotações",
    )
    parser.add_argument(
        "--novo",
        action="store_true",
        help="Ignora checkpoints prévios e inicia uma anotação em branco",
    )
    parser.add_argument(
        "--raio",
        type=int,
        default=1,
        help="Raio em pixels do marcador desenhado na tela",
    )
    parser.add_argument(
        "--tela",
        type=str,
        default=None,
        help="Resolução da tela forçada no formato LARGURAxALTURA (ex: 1920x1080)",
    )
    parser.add_argument(
        "--janela",
        action="store_true",
        help="Abre em modo janela ao invés de tela cheia (Fullscreen)",
    )
    args = parser.parse_args()
    raio_marcador_display = args.raio
    is_fullscreen = not args.janela

    # 1. Determinar resolução do monitor
    if args.tela:
        try:
            sw, sh = map(int, args.tela.lower().split("x"))
            screen_w, screen_h = sw, sh
        except ValueError:
            print(f"[!] Formato inválido para --tela. Usando detecção automática.")
            screen_w, screen_h = obter_resolucao_tela()
    else:
        screen_w, screen_h = obter_resolucao_tela()

    print(f"[*] Resolução da tela detectada para 'Fit': {screen_w}x{screen_h} px")

    # 2. Localização da imagem
    caminho_img = args.imagem
    if not caminho_img.exists():
        fallback_liu = Path("notebooks/liuzywen-RGBTCC/input/DJI_0789_W.JPG")
        fallback_pre = Path("notebooks/DEF-rgbtcc/output/01_pre_transformacao/rgb_preprocessed.jpg")
        if fallback_liu.exists():
            print(f"[!] Usando fallback: {fallback_liu}")
            caminho_img = fallback_liu
        elif fallback_pre.exists():
            print(f"[!] Usando fallback pré-processado: {fallback_pre}")
            caminho_img = fallback_pre
        else:
            print(f"[ERRO] Imagem não encontrada: {caminho_img}")
            sys.exit(1)

    print(f"[*] Carregando imagem original: {caminho_img}")
    img_base = cv2.imread(str(caminho_img))
    if img_base is None:
        print(f"[ERRO] Falha ao decodificar imagem com OpenCV: {caminho_img}")
        sys.exit(1)

    h_orig, w_orig = img_base.shape[:2]
    print(f"[✓] Imagem original carregada: {w_orig}x{h_orig} px")

    caminho_img_ativo = caminho_img
    caminho_saida_ativo = args.saida
    args.saida.mkdir(parents=True, exist_ok=True)

    # 3. Construção da projeção proporcional ('Fit')
    construir_canvas_fit()

    # 4. Continuação automática: Recupera anotações existentes
    stem = caminho_img.stem
    p_csv_stem = args.saida / f"pontos_ground_truth_{stem}.csv"
    p_chk_stem = args.saida / f"checkpoint_{stem}.json"

    if args.carregar:
        n_rec = carregar_anotacoes(args.carregar, img_base.shape)
        if n_rec > 0:
            print(f"[*] Carregadas {n_rec} anotações de: {args.carregar}")
            status_mensagem = f"✓ Carregadas {n_rec} anotações prévias"
            status_cor = (74, 222, 128)
    elif not args.novo:
        arquivo_alvo = p_chk_stem if p_chk_stem.exists() else (p_csv_stem if p_csv_stem.exists() else None)
        if arquivo_alvo:
            n_rec = carregar_anotacoes(arquivo_alvo, img_base.shape)
            if n_rec > 0:
                print(f"[✓ CONTINUAÇÃO AUTOMÁTICA] Recuperadas {n_rec} anotações anteriores de {arquivo_alvo.name}!")
                status_mensagem = f"✓ Recuperados {n_rec} pontos salvos"
                status_cor = (74, 222, 128)

    # 5. Criação da Janela em Fullscreen / Fit
    window_name = "Anotacao de Ground Truth (RGBTCC)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if is_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, screen_w, screen_h)

    cv2.setMouseCallback(window_name, callback_mouse)
    atualizar_canvas()

    print("\n" + "=" * 74)
    print("       ANOTADOR INICIADO EM FULLSCREEN COM 'FIT' À TELA")
    print("=" * 74)
    print(f"  • Resolução da Tela:    {screen_w}x{screen_h} px (Modo Fullscreen)")
    print(f"  • Imagem Original:      {w_orig}x{h_orig} px (Área útil 'Fit': {fit_w}x{fit_h} px)")
    print("  • BOTÃO ESQUERDO:       Marcar ponto (cabeça)")
    print("  • BOTÃO DIREITO / <- :  Desfazer último ponto")
    print("  • BOTÃO [Salvar]:       Salvar Checkpoint (Ctrl+S / S)")
    print("  • BOTÃO [Finalizar]:    Finalizar de fato (F / ESC)")
    print("  • TECLA F11:            Alternar Tela Cheia / Janela")
    print("=" * 74 + "\n")

    # Loop principal de eventos de teclado
    while not deve_encerrar:
        raw_key = cv2.waitKeyEx(30)
        if raw_key == -1:
            continue

        key = raw_key & 0xFF

        # 1. Salvar Checkpoint: Ctrl+S (19), 's'/'S'
        if raw_key in [19, ord("s"), ord("S"), ord("p"), ord("P")] or key in [19, ord("s"), ord("S")]:
            salvar_checkpoint()

        # 2. Finalizar e Encerrar: 'f'/'F', ESC (27), 'q'/'Q'
        elif raw_key in [27, ord("f"), ord("F")] or (key in [ord("f"), ord("F"), ord("q"), ord("Q")] and raw_key not in [65361, 81]):
            print("[*] Comando de finalização acionado pelo teclado.")
            break

        # 3. Desfazer: Ctrl+Z (26), 'z'/'Z', 'u'/'U', Backspace (8), Delete (127), Seta <- (65361 / 81)
        elif raw_key in [26, 65361, 8, 127, 65535, 2424832] or key in [26, ord("z"), ord("Z"), ord("u"), ord("U"), 8, 127]:
            desfazer_ultimo_ponto()

        # 4. Alternar Fullscreen (F11)
        elif raw_key in [65480, 115]:  # F11 keycode
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, screen_w - 100, screen_h - 100)

        # 5. Ajustar tamanho do marcador em tempo real (+ / -)
        elif raw_key in [ord("+"), ord("="), 43, 61]:
            raio_marcador_display = min(8, raio_marcador_display + 1)
            status_mensagem = f"Tamanho do ponto: {raio_marcador_display}px"
            status_cor = (147, 197, 253)
            atualizar_canvas()
        elif raw_key in [ord("-"), ord("_"), 45, 95]:
            raio_marcador_display = max(1, raio_marcador_display - 1)
            status_mensagem = f"Tamanho do ponto: {raio_marcador_display}px"
            status_cor = (147, 197, 253)
            atualizar_canvas()

        # 6. Limpar marcações ('c' / 'C')
        elif key in [ord("c"), ord("C")]:
            if coordenadas:
                print("[!] Limpando todas as marcações.")
                coordenadas.clear()
                status_mensagem = "Todas as anotações foram limpas"
                status_cor = (248, 113, 113)
                atualizar_canvas()

    cv2.destroyAllWindows()

    # --------------------------------------------------------------------------
    # Exportação Final dos Entregáveis (Executada ao Finalizar)
    # --------------------------------------------------------------------------
    p_csv_stem = args.saida / f"pontos_ground_truth_{stem}.csv"
    p_csv_default = args.saida / "pontos_ground_truth.csv"
    p_json_stem = args.saida / f"pontos_ground_truth_{stem}.json"
    p_json_default = args.saida / "pontos_ground_truth.json"
    p_img = args.saida / f"rgb_anotada_ground_truth_{stem}.jpg"

    # 1. Salvar CSV
    df = pd.DataFrame(coordenadas)
    df.to_csv(p_csv_stem, index=False)
    df.to_csv(p_csv_default, index=False)

    # 2. Salvar JSON
    telemetria_gt = {
        "status": "finalizado",
        "data_finalizacao": time.strftime("%Y-%m-%d %H:%M:%S"),
        "imagem_origem": str(caminho_img),
        "arquivo_nome": caminho_img.name,
        "resolucao_original": {
            "largura": int(img_base.shape[1]),
            "altura": int(img_base.shape[0]),
            "canais": int(img_base.shape[2]) if len(img_base.shape) > 2 else 1,
        },
        "resolucao_tela_anotacao": {
            "largura": screen_w,
            "altura": screen_h,
        },
        "total_pessoas_anotadas": len(coordenadas),
        "pontos": coordenadas,
    }
    with open(p_json_stem, "w", encoding="utf-8") as f:
        json.dump(telemetria_gt, f, indent=2, ensure_ascii=False)
    with open(p_json_default, "w", encoding="utf-8") as f:
        json.dump(telemetria_gt, f, indent=2, ensure_ascii=False)

    # 3. Salvar Imagem Anotada em Alta Resolução
    print("[*] Gravando imagem final anotada com os pontos na resolução original...")
    img_anotada_orig = img_base.copy()
    # Raio do ponto na imagem original: proporcional (~10px em 8000x)
    raio_orig = max(4, int(round(w_orig / 800)))
    for pt in coordenadas:
        cv2.circle(img_anotada_orig, (pt["x"], pt["y"]), raio_orig, (0, 0, 255), -1)
        cv2.circle(img_anotada_orig, (pt["x"], pt["y"]), raio_orig + 2, (0, 255, 255), 2)
    cv2.imwrite(str(p_img), img_anotada_orig)

    print("\n" + "=" * 74)
    print("       CONTAGEM FINALIZADA E ENTREGÁVEIS GERADOS COM SUCESSO")
    print("=" * 74)
    print(f"  [✓] Imagem Base Utilizada:     {caminho_img.name} ({w_orig}x{h_orig} px)")
    print(f"  [✓] Total de Pessoas Anotadas: {len(coordenadas)}")
    print(f"  [✓] Tabela CSV de Coordenadas: {p_csv_stem}")
    print(f"  [✓] Metadados e Pontos JSON:   {p_json_stem}")
    print(f"  [✓] Imagem com Auditoria GT:   {p_img}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
