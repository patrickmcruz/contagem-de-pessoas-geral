#!/usr/bin/env python3
"""
Anotador Manual de Pontos (Ground Truth) para Contagem de Pessoas
===============================================================
Permite marcar interativamente a posição das cabeças/pedestres em imagens
RGB pré-processadas do pipeline RGBTCC, gerando os artefatos de Ground Truth
necessários para cálculo de métricas (MAE, MSE) e apresentações.

Recursos:
- Checkpoint / Salvar Progresso: Salva o estado atual sem fechar a janela.
- Auto-Continuação: Ao abrir a mesma imagem, carrega automaticamente os pontos anteriores.
- Finalizar: Encerra a contagem e gera os entregáveis finais completos.
- Desfazer: Remove o último ponto inserido.

Atalhos & Controles:
- Botão Esquerdo (na imagem): Marcar ponto (cabeça)
- Botão [ ↩ Desfazer ] (ou Ctrl+Z / Seta <- / Botão Direito): Desfazer último ponto
- Botão [ 💾 Salvar ] (ou Ctrl+S / S): Salvar progresso atual (checkpoint)
- Botão [ ✓ Finalizar ] (ou F / ESC): Finalizar a contagem e gerar entregáveis
- Tecla C: Limpar todas as marcações
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
raio_ponto = 2
deve_encerrar = False
status_mensagem = "Botao Esq: Marcar | Botao Dir / Ctrl+Z: Desfazer"
status_cor = (203, 213, 225)  # Cinza claro

# Diretório e imagem ativos
caminho_img_ativo = None
caminho_saida_ativo = None

# Coordenadas dos botões no HUD superior
HUD_HEIGHT = 48
BTN_UNDO = (0, 0, 0, 0)
BTN_SAVE = (0, 0, 0, 0)
BTN_FINISH = (0, 0, 0, 0)


def salvar_checkpoint(silencioso: bool = False):
    """Salva o progresso atual em disco (checkpoint) sem fechar a aplicação."""
    global status_mensagem, status_cor
    if caminho_saida_ativo is None:
        return

    caminho_saida_ativo.mkdir(parents=True, exist_ok=True)
    p_csv = caminho_saida_ativo / "pontos_ground_truth.csv"
    p_json = caminho_saida_ativo / "checkpoint_anotacao.json"

    # Salva CSV
    df = pd.DataFrame(coordenadas)
    df.to_csv(p_csv, index=False)

    # Salva JSON de Checkpoint
    checkpoint_data = {
        "tipo": "checkpoint_progresso",
        "data_checkpoint": time.strftime("%Y-%m-%d %H:%M:%S"),
        "imagem_origem": str(caminho_img_ativo),
        "resolucao": {
            "largura": int(img_base.shape[1]),
            "altura": int(img_base.shape[0]),
        },
        "total_pessoas_anotadas": len(coordenadas),
        "pontos": coordenadas,
    }
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    hora_str = time.strftime("%H:%M:%S")
    status_mensagem = f"✓ Salvo as {hora_str} ({len(coordenadas)} pts)"
    status_cor = (74, 222, 128)  # Verde

    if not silencioso:
        print(f"[✓ CHECKPOINT] {len(coordenadas)} anotações salvas em: {p_csv.name} ({hora_str})")
    atualizar_canvas()


def atualizar_canvas():
    """Redesenha todos os pontos sobre a imagem base e projeta o HUD interativo."""
    global img_display, BTN_UNDO, BTN_SAVE, BTN_FINISH
    img_display = img_base.copy()
    h, w = img_base.shape[:2]

    # 1. Desenha os pontos anotados (círculo com borda para contraste)
    for i, pt in enumerate(coordenadas):
        px, py = pt["x"], pt["y"]
        cv2.circle(img_display, (px, py), raio_ponto, (0, 0, 255), -1)       # Vermelho central
        cv2.circle(img_display, (px, py), raio_ponto + 1, (0, 255, 255), 1)  # Borda amarela sutil

    # 2. Faixa do HUD superior (gradiente escuro semi-transparente)
    overlay = img_display.copy()
    cv2.rectangle(overlay, (0, 0), (w, HUD_HEIGHT), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.92, img_display, 0.08, 0, img_display)
    cv2.line(img_display, (0, HUD_HEIGHT), (w, HUD_HEIGHT), (56, 189, 248), 2)

    # 3. Informações à esquerda: Contador de pessoas
    cv2.putText(
        img_display,
        f"Pessoas: {len(coordenadas)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (74, 222, 128),  # Verde
        2,
        cv2.LINE_AA,
    )

    # 4. Status e instruções no centro
    cv2.putText(
        img_display,
        status_mensagem,
        (190, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        status_cor,
        1,
        cv2.LINE_AA,
    )

    # 5. Botões interativos à direita:
    # Botão 1: Desfazer (Laranja/Âmbar)
    u_x1, u_y1, u_x2, u_y2 = w - 465, 8, w - 340, 40
    BTN_UNDO = (u_x1, u_y1, u_x2, u_y2)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (30, 110, 230), -1)
    cv2.rectangle(img_display, (u_x1, u_y1), (u_x2, u_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "<- Desfazer",
        (u_x1 + 14, u_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Botão 2: Salvar Progresso (Azul Celeste)
    s_x1, s_y1, s_x2, s_y2 = w - 330, 8, w - 175, 40
    BTN_SAVE = (s_x1, s_y1, s_x2, s_y2)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (180, 105, 14), -1)
    cv2.rectangle(img_display, (s_x1, s_y1), (s_x2, s_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "Salvar (Ctrl+S)",
        (s_x1 + 10, s_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Botão 3: Finalizar (Verde Esmeralda)
    f_x1, f_y1, f_x2, f_y2 = w - 165, 8, w - 15, 40
    BTN_FINISH = (f_x1, f_y1, f_x2, f_y2)
    cv2.rectangle(img_display, (f_x1, f_y1), (f_x2, f_y2), (40, 150, 60), -1)
    cv2.rectangle(img_display, (f_x1, f_y1), (f_x2, f_y2), (255, 255, 255), 1)
    cv2.putText(
        img_display,
        "V Finalizar",
        (f_x1 + 20, f_y1 + 22),
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
    """Manipula cliques do mouse e botões na tela."""
    global coordenadas, deve_encerrar, status_mensagem, status_cor

    # Botão esquerdo
    if event == cv2.EVENT_LBUTTONDOWN:
        # Verifica se clicou na área do HUD (botões da interface)
        if y <= HUD_HEIGHT:
            u_x1, u_y1, u_x2, u_y2 = BTN_UNDO
            s_x1, s_y1, s_x2, s_y2 = BTN_SAVE
            f_x1, f_y1, f_x2, f_y2 = BTN_FINISH

            # 1. Clicou no botão [ Desfazer ]
            if u_x1 <= x <= u_x2 and u_y1 <= y <= u_y2:
                desfazer_ultimo_ponto()
                return

            # 2. Clicou no botão [ Salvar ] (Checkpoint)
            elif s_x1 <= x <= s_x2 and s_y1 <= y <= s_y2:
                salvar_checkpoint()
                return

            # 3. Clicou no botão [ Finalizar ]
            elif f_x1 <= x <= f_x2 and f_y1 <= y <= f_y2:
                print("[*] Botão 'Finalizar' acionado pelo usuário.")
                deve_encerrar = True
                return
            return

        # Clique dentro da imagem: adiciona um novo ponto
        if 0 <= y < img_base.shape[0] and 0 <= x < img_base.shape[1]:
            novo_id = len(coordenadas) + 1
            coordenadas.append({"id": novo_id, "x": int(x), "y": int(y)})
            status_mensagem = f"Ponto #{novo_id} adicionado (Total: {len(coordenadas)})"
            status_cor = (203, 213, 225)
            print(f"[+] Ponto #{novo_id} anotado em: (x={x}, y={y}) | Total: {len(coordenadas)}")
            atualizar_canvas()

    # Botão direito: desfaz em qualquer lugar da tela
    elif event == cv2.EVENT_RBUTTONDOWN:
        desfazer_ultimo_ponto()


def carregar_anotacoes(caminho_arquivo: Path) -> int:
    """Carrega anotações prévias de um CSV ou JSON para permitir continuação."""
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
                coordenadas = [
                    {"id": i + 1, "x": int(row["x"]), "y": int(row["y"])}
                    for i, row in df.iterrows()
                ]
        return len(coordenadas)
    except Exception as e:
        print(f"[!] Falha ao ler anotações prévias de {caminho_arquivo}: {e}")
        return 0


def main():
    global img_base, raio_ponto, deve_encerrar, caminho_img_ativo, caminho_saida_ativo, status_mensagem, status_cor
    parser = argparse.ArgumentParser(
        description="Anotador Interativo de Pontos de Multidão (Ground Truth) com Checkpoint",
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

    # Configuração de diretórios e caminhos ativos
    caminho_img_ativo = caminho_img
    caminho_saida_ativo = args.saida
    args.saida.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Continuação automática: Recupera anotações existentes (se houver)
    # --------------------------------------------------------------------------
    p_csv_default = args.saida / "pontos_ground_truth.csv"
    p_chk_default = args.saida / "checkpoint_anotacao.json"

    if args.carregar:
        n_rec = carregar_anotacoes(args.carregar)
        if n_rec > 0:
            print(f"[*] Carregadas {n_rec} anotações de: {args.carregar}")
            status_mensagem = f"✓ Carregadas {n_rec} anotacoes prévias"
            status_cor = (74, 222, 128)
    elif not args.novo:
        # Detecta automaticamente se já existe um checkpoint para esta imagem
        arquivo_alvo = p_csv_default if p_csv_default.exists() else (p_chk_default if p_chk_default.exists() else None)
        if arquivo_alvo:
            n_rec = carregar_anotacoes(arquivo_alvo)
            if n_rec > 0:
                print(f"[✓ CONTINUAÇÃO AUTOMÁTICA] Encontrado progresso anterior: {arquivo_alvo.name}")
                print(f"    └─ {n_rec} pessoas recuperadas! Você pode continuar marcando ou desfazer.")
                status_mensagem = f"✓ Recuperados {n_rec} pontos salvos"
                status_cor = (74, 222, 128)

    window_name = "Anotacao de Ground Truth (RGBTCC)"
    flags_win = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 16)
    cv2.namedWindow(window_name, flags_win)
    cv2.resizeWindow(window_name, 1280, 800)
    cv2.setMouseCallback(window_name, callback_mouse)

    atualizar_canvas()

    print("\n" + "=" * 72)
    print("       ANOTADOR INTERATIVO INICIADO COM SUCESSO")
    print("=" * 72)
    print("  • BOTÃO ESQUERDO:    Marcar ponto (ou clicar nos botões do HUD)")
    print("  • BOTÃO DIREITO:     Desfazer último ponto em qualquer lugar")
    print("  • BOTÃO [Salvar]:    Salvar Checkpoint atual (ou aperte Ctrl+S / S)")
    print("  • BOTÃO [Finalizar]: Finalizar e encerrar de fato (ou aperte F / ESC)")
    print("  • TECLA 'C':         Limpar tudo")
    print("=" * 72 + "\n")

    # Loop principal de eventos de teclado com waitKeyEx para teclas especiais
    while not deve_encerrar:
        raw_key = cv2.waitKeyEx(30)
        if raw_key == -1:
            continue

        key = raw_key & 0xFF

        # 1. Salvar Checkpoint:
        # Ctrl+S (ASCII 19), tecla 's' / 'S' (115 / 83), tecla 'p' / 'P' (112 / 80)
        if raw_key in [19, ord("s"), ord("S"), ord("p"), ord("P")] or key in [19, ord("s"), ord("S"), ord("p"), ord("P")]:
            salvar_checkpoint()

        # 2. Finalizar e Encerrar:
        # Tecla 'f' / 'F' (102 / 70), ESC (27), 'q' / 'Q' (113 / 81 se não for seta)
        elif raw_key in [27, ord("f"), ord("F")] or (key in [ord("f"), ord("F"), ord("q"), ord("Q")] and raw_key not in [65361, 81]):
            print("[*] Comando de finalização acionado pelo teclado.")
            break

        # 3. Desfazer (Undo):
        # Ctrl+Z (ASCII 26), 'z'/'Z' (122/90), 'u'/'U' (117/85), Backspace (8), Delete (127), Seta <- (65361 / 81)
        elif raw_key in [26, 65361, 8, 127, 65535, 2424832] or key in [26, ord("z"), ord("Z"), ord("u"), ord("U"), 8, 127]:
            desfazer_ultimo_ponto()

        # 4. Limpar ('c' / 'C')
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
    p_csv = args.saida / "pontos_ground_truth.csv"
    p_json = args.saida / "pontos_ground_truth.json"
    p_img = args.saida / "rgb_anotada_ground_truth.jpg"

    # 1. Salvar CSV
    df = pd.DataFrame(coordenadas)
    df.to_csv(p_csv, index=False)

    # 2. Salvar JSON
    telemetria_gt = {
        "status": "finalizado",
        "data_finalizacao": time.strftime("%Y-%m-%d %H:%M:%S"),
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

    print("\n" + "=" * 72)
    print("       CONTAGEM FINALIZADA E ENTREGÁVEIS GERADOS COM SUCESSO")
    print("=" * 72)
    print(f"  [✓] Total de Pessoas Anotadas: {len(coordenadas)}")
    print(f"  [✓] Tabela CSV de Coordenadas: {p_csv}")
    print(f"  [✓] Metadados e Pontos JSON:   {p_json}")
    print(f"  [✓] Imagem com Auditoria GT:   {p_img}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
