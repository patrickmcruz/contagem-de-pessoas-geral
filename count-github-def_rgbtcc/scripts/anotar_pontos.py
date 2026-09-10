#!/usr/bin/env python3
"""
Anotador Manual de Pontos (Ground Truth) para Contagem de Pessoas
===============================================================
Permite inspecionar a imagem original de altíssima resolução (8000x6000 px)
com navegação interativa (Zoom e Panorâmica / Pan) em tempo real, permitindo
identificar visualmente com máxima nitidez cada cabeça individual na multidão.

Controles de Navegação em Alta Resolução:
- Roda do Mouse (Scroll): Zoom In / Zoom Out centralizado no cursor do mouse.
- Botão do Meio (Scroll Click) e Arrastar: Mover pela cena (Panorâmica / Pan).
- Segurar Tecla ESPAÇO + Botão Esquerdo: Mover pela cena (Pan como no Photoshop/Figma).
- Teclas W, A, S, D ou Setas: Mover a visualização pela imagem.
- Tecla R ou Botão [ ⟲ Fit ]: Resetar para a visão geral da imagem inteira.
- Teclas I / O ou Botões [ + ] e [ - ]: Zoom In / Zoom Out.
- Mini-Mapa (Picture-in-Picture): No canto inferior direito indicando a região visível.

Controles de Anotação:
- Botão Esquerdo: Marcar ponto (cabeça) com precisão cirúrgica na resolução nativa.
- Botão [ <- Desfazer ] (ou Ctrl+Z / Seta <- / Botão Direito): Desfazer último ponto.
- Botão [ 💾 Salvar ] (ou Ctrl+S / S): Salvar progresso atual (checkpoint).
- Botão [ ✓ Finalizar ] (ou F / ESC): Finalizar a contagem e gerar entregáveis.
- Teclas [ + ] e [ - ]: Ajustar tamanho do marcador na tela.
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
img_thumb = None            # Miniatura para o mini-mapa PiP
img_display = None          # Frame final renderizado para cv2.imshow
raio_marcador_display = 2   # Raio na tela do monitor
is_fullscreen = True

# Estado de Zoom e Navegação (Pan)
zoom_level = 1.0            # Escala de zoom (1.0 = Fit tela, até 12.0x)
center_x = 4000.0           # Posição central X na imagem 8000x6000
center_y = 3000.0           # Posição central Y na imagem 8000x6000
is_dragging_pan = False
pan_start_screen = (0, 0)
pan_start_center = (0.0, 0.0)
space_is_pressed = False

# Projeção da janela
screen_w = 1920
screen_h = 1080
HUD_HEIGHT = 50

# Coordenadas do viewport atual na imagem 8000x6000
roi_x1, roi_y1, roi_x2, roi_y2 = 0.0, 0.0, 8000.0, 6000.0
fit_w, fit_h = 1371, 1028
offset_x, offset_y = 274, 50
scale_fit = 0.1713

deve_encerrar = False
status_mensagem = "Rolar mouse: Zoom | Meio/Espaco: Arrastar | R: Reset Fit | [+/-]: Marcador"
status_cor = (203, 213, 225)

# Diretório e imagem ativos
caminho_img_ativo = None
caminho_saida_ativo = None

# Botões interativos no HUD
BTN_ZOOM_IN = (0, 0, 0, 0)
BTN_ZOOM_OUT = (0, 0, 0, 0)
BTN_RESET_ZOOM = (0, 0, 0, 0)
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


def calcular_roi():
    """Calcula a janela visível (ROI) na imagem original 8000x6000 baseada no zoom e centro."""
    global roi_x1, roi_y1, roi_x2, roi_y2, center_x, center_y
    h_orig, w_orig = img_base.shape[:2]

    # Dimensão da janela na imagem original
    roi_w = w_orig / zoom_level
    roi_h = h_orig / zoom_level

    # Coordenadas com centro desejado
    x1 = center_x - roi_w / 2.0
    y1 = center_y - roi_h / 2.0
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    # Clamping dentro das bordas da imagem original
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if x2 > w_orig:
        x1 -= (x2 - w_orig)
        x2 = float(w_orig)
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if y2 > h_orig:
        y1 -= (y2 - h_orig)
        y2 = float(h_orig)

    roi_x1 = max(0.0, x1)
    roi_y1 = max(0.0, y1)
    roi_x2 = min(float(w_orig), x2)
    roi_y2 = min(float(h_orig), y2)

    center_x = (roi_x1 + roi_x2) / 2.0
    center_y = (roi_y1 + roi_y2) / 2.0


def aplicar_zoom(fator: float, cursor_screen_x: int = None, cursor_screen_y: int = None):
    """Aplica zoom in ou zoom out centralizado no cursor do mouse ou no centro atual."""
    global zoom_level, center_x, center_y, status_mensagem, status_cor

    novo_zoom = np.clip(zoom_level * fator, 1.0, 12.0)
    if abs(novo_zoom - zoom_level) < 0.01:
        return

    # Se uma posição na tela foi fornecida, centraliza o zoom no ponto sob o mouse
    if cursor_screen_x is not None and cursor_screen_y is not None:
        if offset_x <= cursor_screen_x < offset_x + fit_w and offset_y <= cursor_screen_y < offset_y + fit_h:
            norm_x = (cursor_screen_x - offset_x) / fit_w
            norm_y = (cursor_screen_y - offset_y) / fit_h
            orig_mouse_x = roi_x1 + norm_x * (roi_x2 - roi_x1)
            orig_mouse_y = roi_y1 + norm_y * (roi_y2 - roi_y1)

            # Novo centro para manter o ponto do mouse fixo
            center_x = orig_mouse_x - (norm_x - 0.5) * (img_base.shape[1] / novo_zoom)
            center_y = orig_mouse_y - (norm_y - 0.5) * (img_base.shape[0] / novo_zoom)

    zoom_level = novo_zoom
    status_mensagem = f"Zoom: {zoom_level:.1f}x | Rolar mouse para aproximar/afastar"
    status_cor = (147, 197, 253)
    atualizar_canvas()


def reset_zoom():
    """Reseta a visualização para o enquadramento completo (Fit)."""
    global zoom_level, center_x, center_y, status_mensagem, status_cor
    h_orig, w_orig = img_base.shape[:2]
    zoom_level = 1.0
    center_x = w_orig / 2.0
    center_y = h_orig / 2.0
    status_mensagem = "Visão geral da imagem (Fit completo)"
    status_cor = (203, 213, 225)
    atualizar_canvas()


def mover_pan(dx_screen: int, dy_screen: int):
    """Move a câmera pela cena com base no deslocamento na tela."""
    global center_x, center_y
    if fit_w <= 0 or fit_h <= 0:
        return
    delta_orig_x = (dx_screen / fit_w) * (roi_x2 - roi_x1)
    delta_orig_y = (dy_screen / fit_h) * (roi_y2 - roi_y1)

    center_x -= delta_orig_x
    center_y -= delta_orig_y
    atualizar_canvas()


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


def atualizar_canvas():
    """Renderiza a região visível em alta resolução, projeta os pontos e o HUD."""
    global img_display, BTN_ZOOM_IN, BTN_ZOOM_OUT, BTN_RESET_ZOOM, BTN_UNDO, BTN_SAVE, BTN_FINISH
    global fit_w, fit_h, offset_x, offset_y, scale_fit

    calcular_roi()

    h_orig, w_orig = img_base.shape[:2]
    w = screen_w
    avail_w = screen_w
    avail_h = screen_h - HUD_HEIGHT

    # Recorte da região de interesse na imagem original 8000x6000
    rx1, ry1 = int(round(roi_x1)), int(round(roi_y1))
    rx2, ry2 = int(round(roi_x2)), int(round(roi_y2))
    crop = img_base[ry1:ry2, rx1:rx2]

    # Proporção do crop para a tela
    crop_h, crop_w = crop.shape[:2]
    scale_fit = min(avail_w / max(1, crop_w), avail_h / max(1, crop_h))
    fit_w = int(round(crop_w * scale_fit))
    fit_h = int(round(crop_h * scale_fit))

    offset_x = (avail_w - fit_w) // 2
    offset_y = HUD_HEIGHT + (avail_h - fit_h) // 2

    # Canvas principal escuro
    img_display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    img_display[:] = (18, 22, 30)

    # Redimensionamento rápido e nítido do crop para a tela
    interp = cv2.INTER_LINEAR if zoom_level > 2.0 else cv2.INTER_AREA
    crop_disp = cv2.resize(crop, (fit_w, fit_h), interpolation=interp)
    img_display[offset_y:offset_y + fit_h, offset_x:offset_x + fit_w] = crop_disp

    # Moldura sutil ao redor da área de visualização
    cv2.rectangle(
        img_display,
        (offset_x - 1, offset_y - 1),
        (offset_x + fit_w, offset_y + fit_h),
        (51, 65, 85),
        1,
    )

    # 1. Desenha os pontos anotados que caem dentro da ROI visível
    for pt in coordenadas:
        px, py = pt["x"], pt["y"]
        if roi_x1 <= px <= roi_x2 and roi_y1 <= py <= roi_y2:
            norm_x = (px - roi_x1) / (roi_x2 - roi_x1)
            norm_y = (py - roi_y1) / (roi_y2 - roi_y1)
            disp_x = int(round(offset_x + norm_x * fit_w))
            disp_y = int(round(offset_y + norm_y * fit_h))

            # Desenha marcador
            if raio_marcador_display <= 1:
                cv2.circle(img_display, (disp_x, disp_y), 1, (0, 0, 255), -1)
            else:
                cv2.circle(img_display, (disp_x, disp_y), raio_marcador_display, (0, 0, 255), -1)
                cv2.circle(img_display, (disp_x, disp_y), raio_marcador_display + 1, (0, 255, 255), 1)

    # 2. Mini-Mapa PiP (Picture-in-Picture) no canto inferior direito quando ampliado
    if zoom_level > 1.05 and img_thumb is not None:
        th_h, th_w = img_thumb.shape[:2]
        pip_x = screen_w - th_w - 20
        pip_y = screen_h - th_h - 20
        img_display[pip_y:pip_y + th_h, pip_x:pip_x + th_w] = img_thumb
        cv2.rectangle(img_display, (pip_x - 1, pip_y - 1), (pip_x + th_w, pip_y + th_h), (200, 200, 200), 1)

        # Retângulo indicativo da visualização atual
        rect_x1 = pip_x + int(round(roi_x1 / w_orig * th_w))
        rect_y1 = pip_y + int(round(roi_y1 / h_orig * th_h))
        rect_x2 = pip_x + int(round(roi_x2 / w_orig * th_w))
        rect_y2 = pip_y + int(round(roi_y2 / h_orig * th_h))
        cv2.rectangle(img_display, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 255), 2)

    # 3. Barra de HUD superior
    cv2.rectangle(img_display, (0, 0), (w, HUD_HEIGHT), (15, 23, 42), -1)
    cv2.line(img_display, (0, HUD_HEIGHT), (w, HUD_HEIGHT), (56, 189, 248), 2)

    # Placar à esquerda
    cv2.putText(
        img_display,
        f"Pessoas: {len(coordenadas)} | Zoom: {zoom_level:.1f}x",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (74, 222, 128),  # Verde
        2,
        cv2.LINE_AA,
    )

    # Status e instruções no centro
    cv2.putText(
        img_display,
        status_mensagem,
        (350, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        status_cor,
        1,
        cv2.LINE_AA,
    )

    # Botões interativos à direita:
    # Botão: Zoom +
    z1_x1, z1_y1, z1_x2, z1_y2 = w - 690, 8, w - 645, 42
    BTN_ZOOM_IN = (z1_x1, z1_y1, z1_x2, z1_y2)
    cv2.rectangle(img_display, (z1_x1, z1_y1), (z1_x2, z1_y2), (30, 41, 59), -1)
    cv2.rectangle(img_display, (z1_x1, z1_y1), (z1_x2, z1_y2), (148, 163, 184), 1)
    cv2.putText(img_display, "+", (z1_x1 + 14, z1_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Botão: Zoom -
    z2_x1, z2_y1, z2_x2, z2_y2 = w - 635, 8, w - 590, 42
    BTN_ZOOM_OUT = (z2_x1, z2_y1, z2_x2, z2_y2)
    cv2.rectangle(img_display, (z2_x1, z2_y1), (z2_x2, z2_y2), (30, 41, 59), -1)
    cv2.rectangle(img_display, (z2_x1, z2_y1), (z2_x2, z2_y2), (148, 163, 184), 1)
    cv2.putText(img_display, "-", (z2_x1 + 16, z2_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Botão: Reset Zoom (Fit)
    rz_x1, rz_y1, rz_x2, rz_y2 = w - 580, 8, w - 495, 42
    BTN_RESET_ZOOM = (rz_x1, rz_y1, rz_x2, rz_y2)
    cv2.rectangle(img_display, (rz_x1, rz_y1), (rz_x2, rz_y2), (30, 41, 59), -1)
    cv2.rectangle(img_display, (rz_x1, rz_y1), (rz_x2, rz_y2), (148, 163, 184), 1)
    cv2.putText(img_display, "Fit (R)", (rz_x1 + 12, rz_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Botão: Desfazer
    u_x1, u_y1, u_x2, u_y2 = w - 485, 8, w - 365, 42
    BTN_UNDO = (u_x1, u_y1, u_x2, u_y2)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (30, 110, 230), -1)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (255, 255, 255), 1)
    cv2.putText(img_display, "<- Desfazer", (u_x1 + 12, u_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Botão: Salvar
    s_x1, s_y1, s_x2, s_y2 = w - 355, 8, w - 195, 42
    BTN_SAVE = (s_x1, s_y1, s_x2, s_y2)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (180, 105, 14), -1)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (255, 255, 255), 1)
    cv2.putText(img_display, "Salvar (Ctrl+S)", (s_x1 + 10, s_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Botão: Finalizar
    f_x1, f_y1, f_x2, f_y2 = w - 185, 8, w - 15, 42
    BTN_FINISH = (f_x1, f_y1, f_x2, f_y2)
    cv2.rectangle(img_display, (f_x1, f_y1), (f_x2, f_y2), (40, 150, 60), -1)
    cv2.rectangle(img_display, (f_x1, f_y1), (f_x2, f_y2), (255, 255, 255), 1)
    cv2.putText(img_display, "V Finalizar", (f_x1 + 16, f_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

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
    """Manipula eventos do mouse: Zoom por scroll, Pan por arraste e marcação de pontos."""
    global coordenadas, deve_encerrar, status_mensagem, status_cor
    global is_dragging_pan, pan_start_screen, pan_start_center, center_x, center_y

    # 1. Roda do Mouse (Zoom In / Out no cursor)
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            aplicar_zoom(1.35, x, y)
        else:
            aplicar_zoom(1.0 / 1.35, x, y)
        return

    # 2. Iniciar Pan (Botão do Meio ou Segurar Espaço + Botão Esquerdo)
    if event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and space_is_pressed):
        is_dragging_pan = True
        pan_start_screen = (x, y)
        pan_start_center = (center_x, center_y)
        return

    # 3. Arrastando (Pan em andamento)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging_pan:
            dx = x - pan_start_screen[0]
            dy = y - pan_start_screen[1]
            if fit_w > 0 and fit_h > 0:
                delta_orig_x = (dx / fit_w) * (roi_x2 - roi_x1)
                delta_orig_y = (dy / fit_h) * (roi_y2 - roi_y1)
                center_x = pan_start_center[0] - delta_orig_x
                center_y = pan_start_center[1] - delta_orig_y
                atualizar_canvas()
            return

    # 4. Soltar Pan
    elif event == cv2.EVENT_MBUTTONUP or (event == cv2.EVENT_LBUTTONUP and is_dragging_pan):
        is_dragging_pan = False
        return

    # 5. Botão Esquerdo (Clique normal)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Clique no HUD
        if y <= HUD_HEIGHT:
            if BTN_ZOOM_IN[0] <= x <= BTN_ZOOM_IN[2] and BTN_ZOOM_IN[1] <= y <= BTN_ZOOM_IN[3]:
                aplicar_zoom(1.4)
                return
            elif BTN_ZOOM_OUT[0] <= x <= BTN_ZOOM_OUT[2] and BTN_ZOOM_OUT[1] <= y <= BTN_ZOOM_OUT[3]:
                aplicar_zoom(1.0 / 1.4)
                return
            elif BTN_RESET_ZOOM[0] <= x <= BTN_RESET_ZOOM[2] and BTN_RESET_ZOOM[1] <= y <= BTN_RESET_ZOOM[3]:
                reset_zoom()
                return
            elif BTN_UNDO[0] <= x <= BTN_UNDO[2] and BTN_UNDO[1] <= y <= BTN_UNDO[3]:
                desfazer_ultimo_ponto()
                return
            elif BTN_SAVE[0] <= x <= BTN_SAVE[2] and BTN_SAVE[1] <= y <= BTN_SAVE[3]:
                salvar_checkpoint()
                return
            elif BTN_FINISH[0] <= x <= BTN_FINISH[2] and BTN_FINISH[1] <= y <= BTN_FINISH[3]:
                print("[*] Botão 'Finalizar' acionado.")
                deve_encerrar = True
                return
            return

        # Clique dentro da imagem: Marcação de ponto na resolução original
        if offset_x <= x < offset_x + fit_w and offset_y <= y < offset_y + fit_h:
            h_orig, w_orig = img_base.shape[:2]
            norm_x = (x - offset_x) / fit_w
            norm_y = (y - offset_y) / fit_h

            orig_x = int(round(roi_x1 + norm_x * (roi_x2 - roi_x1)))
            orig_y = int(round(roi_y1 + norm_y * (roi_y2 - roi_y1)))

            orig_x = max(0, min(w_orig - 1, orig_x))
            orig_y = max(0, min(h_orig - 1, orig_y))

            novo_id = len(coordenadas) + 1
            coordenadas.append({"id": novo_id, "x": orig_x, "y": orig_y})
            status_mensagem = f"Ponto #{novo_id} anotado em ({orig_x}, {orig_y}) | Total: {len(coordenadas)}"
            status_cor = (203, 213, 225)
            print(f"[+] Ponto #{novo_id} anotado: Tela=({x}, {y}) -> Original=({orig_x}, {orig_y}) | Total: {len(coordenadas)}")
            atualizar_canvas()

    # 6. Botão Direito: Desfazer
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
    global img_base, img_thumb, raio_marcador_display, deve_encerrar, caminho_img_ativo, caminho_saida_ativo
    global screen_w, screen_h, is_fullscreen, status_mensagem, status_cor, space_is_pressed
    global center_x, center_y, zoom_level

    parser = argparse.ArgumentParser(
        description="Anotador Interativo em Alta Resolução com Zoom e Pan para Imagens 8000x",
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
        default=2,
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

    print(f"[*] Resolução da tela detectada: {screen_w}x{screen_h} px")

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

    print(f"[*] Carregando imagem de alta resolução: {caminho_img}")
    img_base = cv2.imread(str(caminho_img))
    if img_base is None:
        print(f"[ERRO] Falha ao decodificar imagem com OpenCV: {caminho_img}")
        sys.exit(1)

    h_orig, w_orig = img_base.shape[:2]
    center_x = w_orig / 2.0
    center_y = h_orig / 2.0
    print(f"[✓] Imagem de altíssima resolução carregada: {w_orig}x{h_orig} px")

    # Pré-computar miniatura para o Mini-Mapa PiP
    thumb_w = 180
    thumb_h = int(round(180 * (h_orig / w_orig)))
    img_thumb = cv2.resize(img_base, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    caminho_img_ativo = caminho_img
    caminho_saida_ativo = args.saida
    args.saida.mkdir(parents=True, exist_ok=True)

    # 3. Continuação automática: Recupera anotações existentes
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

    # 4. Criação da Janela
    window_name = "Anotacao de Ground Truth (RGBTCC)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if is_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, screen_w, screen_h)

    cv2.setMouseCallback(window_name, callback_mouse)
    atualizar_canvas()

    print("\n" + "=" * 76)
    print(f"   ANOTADOR INICIADO EM ALTA RESOLUÇÃO COM ZOOM E PAN (8000x6000 px)")
    print("=" * 76)
    print("  • ROLAR MOUSE (Scroll):  Zoom In / Zoom Out exato no local apontado")
    print("  • BOTÃO DO MEIO ARRASTAR:Mover (Pan) suavemente pela cena")
    print("  • SEGURAR ESPAÇO + ESQ:  Mover (Pan) pela cena (estilo Photoshop/Figma)")
    print("  • TECLAS W / A / S / D:  Mover a câmera para cima/esquerda/baixo/direita")
    print("  • TECLA 'R' / [Fit]:     Resetar o Zoom (visão geral 100% da imagem)")
    print("  • BOTÃO ESQUERDO:        Marcar cabeça na resolução máxima com precisão")
    print("  • BOTÃO DIREITO / <- :   Desfazer último ponto")
    print("  • BOTÃO [Salvar]:        Salvar Checkpoint atual (Ctrl+S / S)")
    print("  • BOTÃO [Finalizar]:     Finalizar e encerrar de fato (F / ESC)")
    print("  • TECLAS [ + ] e [ - ]:  Ajustar tamanho do ponto visual na tela")
    print("  • TECLA F11:             Alternar Tela Cheia / Janela")
    print("=" * 76 + "\n")

    # Loop principal de eventos de teclado
    step_pan = 60  # Pixels de pan no teclado
    while not deve_encerrar:
        raw_key = cv2.waitKeyEx(30)
        if raw_key == -1:
            space_is_pressed = False
            continue

        key = raw_key & 0xFF

        # Detecta barra de espaço para pan com o mouse
        if key == 32:
            space_is_pressed = True

        # 1. Salvar Checkpoint: Ctrl+S (19), tecla 's' / 'S' (quando não em pan)
        if raw_key == 19 or key in [ord("s"), ord("S")]:
            salvar_checkpoint()

        # 2. Finalizar e Encerrar: 'f' / 'F', ESC (27), 'q' / 'Q'
        elif raw_key in [27, ord("f"), ord("F")] or (key in [ord("f"), ord("F"), ord("q"), ord("Q")] and raw_key not in [65361, 81]):
            print("[*] Comando de finalização acionado pelo teclado.")
            break

        # 3. Desfazer: Ctrl+Z (26), 'z'/'Z', 'u'/'U', Backspace (8), Delete (127), Seta <- (65361 / 81)
        elif raw_key in [26, 65361, 8, 127, 65535, 2424832] or key in [26, ord("z"), ord("Z"), ord("u"), ord("U"), 8, 127]:
            desfazer_ultimo_ponto()

        # 4. Zoom pelo teclado: 'i'/'I' (Zoom in), 'o'/'O' (Zoom out), 'r'/'R' (Reset)
        elif key in [ord("i"), ord("I")]:
            aplicar_zoom(1.35)
        elif key in [ord("o"), ord("O")]:
            aplicar_zoom(1.0 / 1.35)
        elif key in [ord("r"), ord("R")]:
            reset_zoom()

        # 5. Pan pelo teclado: W (cima), A (esquerda), D (direita), Seta Cima/Baixo/Esq/Dir
        elif key in [ord("w"), ord("W")] or raw_key in [65362, 82]:  # Up
            mover_pan(0, step_pan)
        elif key in [ord("a"), ord("A")]:  # Left
            mover_pan(step_pan, 0)
        elif key in [ord("d"), ord("D")] or raw_key in [65363, 83]:  # Right
            mover_pan(-step_pan, 0)
        elif raw_key in [65364, 84]:  # Down
            mover_pan(0, -step_pan)

        # 6. Ajustar tamanho do marcador (+ / -)
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

        # 7. Alternar Fullscreen (F11)
        elif raw_key in [65480, 115]:
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, screen_w - 100, screen_h - 100)

        # 8. Limpar marcações ('c' / 'C')
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
        "total_pessoas_anotadas": len(coordenadas),
        "pontos": coordenadas,
    }
    with open(p_json_stem, "w", encoding="utf-8") as f:
        json.dump(telemetria_gt, f, indent=2, ensure_ascii=False)
    with open(p_json_default, "w", encoding="utf-8") as f:
        json.dump(telemetria_gt, f, indent=2, ensure_ascii=False)

    # 3. Salvar Imagem Anotada em Alta Resolução
    print("[*] Gravando imagem final anotada com os pontos na resolução original de 8000x6000...")
    img_anotada_orig = img_base.copy()
    raio_orig = max(4, int(round(w_orig / 800)))
    for pt in coordenadas:
        cv2.circle(img_anotada_orig, (pt["x"], pt["y"]), raio_orig, (0, 0, 255), -1)
        cv2.circle(img_anotada_orig, (pt["x"], pt["y"]), raio_orig + 2, (0, 255, 255), 2)
    cv2.imwrite(str(p_img), img_anotada_orig)

    print("\n" + "=" * 76)
    print("       CONTAGEM FINALIZADA E ENTREGÁVEIS GERADOS COM SUCESSO")
    print("=" * 76)
    print(f"  [✓] Imagem Base Utilizada:     {caminho_img.name} ({w_orig}x{h_orig} px)")
    print(f"  [✓] Total de Pessoas Anotadas: {len(coordenadas)}")
    print(f"  [✓] Tabela CSV de Coordenadas: {p_csv_stem}")
    print(f"  [✓] Metadados e Pontos JSON:   {p_json_stem}")
    print(f"  [✓] Imagem com Auditoria GT:   {p_img}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
