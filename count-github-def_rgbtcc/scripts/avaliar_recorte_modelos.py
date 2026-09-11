#!/usr/bin/env python3
"""
Avaliação Rápida de Modelos RGBT-CC em Recortes com Poucas Pessoas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Este script extrai recortes específicos do par multimodal alinhado
(regiões com 0, 5, 9 e 19 pessoas) e executa a inferência dos modelos
(DualStreamRGBTNet / DEF-rgbtcc e LiuzywenRGBTCCNet) para analisar
como modelos baseados em regressão de mapa de densidade se comportam
em cenários de densidade esparsa / poucas pessoas.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Configurar diretório de cache do Matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Definir diretórios base
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NOTEBOOK_DIR = ROOT_DIR / "notebooks" / "DEF-rgbtcc"
OUTPUT_DIR = NOTEBOOK_DIR / "output" / "recorte_poucas_pessoas"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Carregar modelo DEF-rgbtcc
sys.path.insert(0, str(NOTEBOOK_DIR))
from models.models import DualStreamRGBTNet

# Descarregar completamente o pacote 'models' para evitar colisões
for k in list(sys.modules.keys()):
    if k == "models" or k.startswith("models."):
        del sys.modules[k]
sys.path.pop(0)

# 2. Carregar modelo liuzywen-RGBTCC
sys.path.insert(0, str(ROOT_DIR / "notebooks" / "liuzywen-RGBTCC"))
try:
    from models import build_model as build_liuzywen_model
    LIUZYWEN_AVAILABLE = True
except Exception as e:
    print(f"[!] Aviso: Modelo Liuzywen não pôde ser importado: {e}")
    LIUZYWEN_AVAILABLE = False


def carregar_ground_truth(w_raw=8000, h_raw=6000):
    """Carrega os pontos do ground truth e projeta para o espaço 1280x1024."""
    gt_file = NOTEBOOK_DIR / "output" / "ground_truth" / "pontos_ground_truth_DJI_0789_W.json"
    if not gt_file.exists():
        print(f"[!] Ground truth não encontrado em {gt_file}")
        return []

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = data.get("pontos", [])
    crop_h = int(h_raw * 0.70)
    crop_w = int(crop_h * 1.25)
    left = (w_raw - crop_w) // 2
    top = (h_raw - crop_h) // 2
    scale_x = 1280 / crop_w
    scale_y = 1024 / crop_h
    shift_x, shift_y = -22, -23

    aligned_pts = []
    for p in pts:
        x_raw, y_raw = p["x"], p["y"]
        if left <= x_raw < left + crop_w and top <= y_raw < top + crop_h:
            x_c = x_raw - left
            y_c = y_raw - top
            x_al = int(round(x_c * scale_x + shift_x))
            y_al = int(round(y_c * scale_y + shift_y))
            if 0 <= x_al < 1280 and 0 <= y_al < 1024:
                aligned_pts.append({"id": p["id"], "x": x_al, "y": y_al})

    return aligned_pts


def avaliar_recorte(
    img_rgb: np.ndarray,
    img_th: np.ndarray,
    aligned_pts: list,
    roi: tuple,
    nome_roi: str,
    model_def,
    model_liu,
    device: torch.device,
):
    """Executa a contagem e auditoria visual em um recorte específico."""
    x1, x2, y1, y2 = roi
    crop_rgb = img_rgb[y1:y2, x1:x2].copy()
    crop_th = img_th[y1:y2, x1:x2].copy()

    # Pontos de ground truth dentro da ROI
    pts_roi = [p for p in aligned_pts if x1 <= p["x"] < x2 and y1 <= p["y"] < y2]
    real_count = len(pts_roi)

    # Coordenadas relativas ao recorte
    pts_rel = [{"x": p["x"] - x1, "y": p["y"] - y1} for p in pts_roi]

    # Ajustar para múltiplo de 32 para a rede neural
    ch, cw = crop_rgb.shape[:2]
    cw_32 = ((cw + 31) // 32) * 32
    ch_32 = ((ch + 31) // 32) * 32

    crop_rgb_res = cv2.resize(crop_rgb, (cw_32, ch_32))
    crop_th_res = cv2.resize(crop_th, (cw_32, ch_32))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t_rgb = transform(crop_rgb_res).unsqueeze(0).to(device)
    t_th = transform(crop_th_res).unsqueeze(0).to(device)

    # 1. Inferência DEF-rgbtcc (DualStreamRGBTNet)
    count_def = 0.0
    dmap_def_vis = None
    if model_def is not None:
        with torch.no_grad():
            out_def = model_def(t_rgb, t_th)
            dmap_def = out_def.squeeze().cpu().numpy()
            dmap_def = np.clip(dmap_def, 0, None)
            count_def = float(np.sum(dmap_def))
            dmap_def_vis = cv2.resize(dmap_def, (cw, ch), interpolation=cv2.INTER_CUBIC)

    # 2. Inferência Liuzywen
    count_liu = 0.0
    token_liu = 0.0
    dmap_liu_vis = None
    if model_liu is not None:
        with torch.no_grad():
            out_liu = model_liu(t_rgb, t_th)
            dmap_liu = out_liu["density_map"].squeeze().cpu().numpy()
            dmap_liu = np.clip(dmap_liu, 0, None)
            count_liu = float(out_liu.get("count", torch.tensor(np.sum(dmap_liu))).item())
            token_liu = float(out_liu.get("token_count", torch.tensor(0.0)).item())
            dmap_liu_vis = cv2.resize(dmap_liu, (cw, ch), interpolation=cv2.INTER_CUBIC)

    # 3. Análise de Picos Locais (Detecção discreta de cabeças nos mapas de densidade)
    from scipy.ndimage import maximum_filter
    thresh = max(0.003, 0.15 * dmap_def.max()) if dmap_def is not None else 0.01
    local_max = (maximum_filter(dmap_def, size=9) == dmap_def) & (dmap_def > thresh)
    picos_detectados = int(np.sum(local_max)) if dmap_def is not None else 0

    # Gerar imagem RGB com Ground Truth desenhado
    vis_gt = crop_rgb.copy()
    for pt in pts_rel:
        cv2.circle(vis_gt, (pt["x"], pt["y"]), 5, (0, 255, 0), -1)
        cv2.circle(vis_gt, (pt["x"], pt["y"]), 6, (0, 0, 255), 1)

    # Gerar Heatmap e Projeção Sobreposta
    if dmap_def_vis is not None:
        # Remover artefato de borda extrema se houver
        d_clean = dmap_def_vis.copy()
        # Normalização robusta baseada no percentil 99 para destacar os pedestres
        p99 = np.percentile(d_clean, 99.2)
        vmax = max(p99, 0.002)
        d_norm = np.clip(d_clean / vmax, 0.0, 1.0)
        d_norm_uint8 = (d_norm * 255).astype(np.uint8)

        heat_color = cv2.applyColorMap(d_norm_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        overlay_def = cv2.addWeighted(crop_rgb, 0.60, heat_color, 0.40, 0)
    else:
        heat_color = np.zeros_like(crop_rgb)
        overlay_def = crop_rgb

    # Montar painel visual comparativo em 5 colunas
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(crop_rgb)
    axes[0].set_title(f"1. Recorte RGB ({cw}x{ch} px)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(crop_th)
    axes[1].set_title("2. Recorte Térmico LWIR", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(vis_gt)
    axes[2].set_title(f"3. Ground Truth ({real_count} reais)", fontsize=11, fontweight="bold", color="darkgreen")
    axes[2].axis("off")

    axes[3].imshow(heat_color)
    axes[3].set_title(f"4. Mapa de Densidade 2D", fontsize=11, fontweight="bold", color="darkorange")
    axes[3].axis("off")

    axes[4].imshow(overlay_def)
    axes[4].set_title(f"5. Sobreposição (Previsto: {count_def:.1f})", fontsize=11, fontweight="bold", color="darkred")
    axes[4].axis("off")

    plt.tight_layout()
    slug = nome_roi.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
    painel_path = OUTPUT_DIR / f"painel_{slug}.jpg"
    plt.savefig(str(painel_path), dpi=150, bbox_inches="tight")
    plt.close()

    # Salvar recortes brutos
    cv2.imwrite(str(OUTPUT_DIR / f"{slug}_rgb.jpg"), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / f"{slug}_thermal.jpg"), cv2.cvtColor(crop_th, cv2.COLOR_RGB2BGR))

    return {
        "nome": nome_roi,
        "coordenadas_roi": {"x1": x1, "x2": x2, "y1": y1, "y2": y2},
        "dimensoes_crop": {"largura": cw, "altura": ch},
        "pessoas_reais_ground_truth": real_count,
        "modelo_def_rgbtcc_estimado": round(count_def, 2),
        "modelo_def_rgbtcc_picos_locais": picos_detectados,
        "modelo_liuzywen_estimado": round(count_liu, 2),
        "modelo_liuzywen_token": round(token_liu, 2),
        "painel_auditoria": str(painel_path.relative_to(ROOT_DIR)),
    }


def main():
    parser = argparse.ArgumentParser(description="Avaliar modelos em recortes com poucas pessoas.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[*] Dispositivo de inferência: {device}")

    # Carregar imagens de entrada
    p_rgb = NOTEBOOK_DIR / "output" / "01_pre_transformacao" / "rgb_preprocessed.jpg"
    p_th = NOTEBOOK_DIR / "output" / "01_pre_transformacao" / "thermal_preprocessed.jpg"

    if not p_rgb.exists() or not p_th.exists():
        print(f"[!] Insumos do estágio 1 não encontrados. Execute o Notebook 01 primeiro.")
        sys.exit(1)

    img_rgb = cv2.cvtColor(cv2.imread(str(p_rgb)), cv2.COLOR_BGR2RGB)
    img_th = cv2.cvtColor(cv2.imread(str(p_th)), cv2.COLOR_BGR2RGB)
    print(f"[✓] Imagens multimodais carregadas ({img_rgb.shape[1]}x{img_rgb.shape[0]} px)")

    # Carregar ground truth
    aligned_pts = carregar_ground_truth()
    print(f"[✓] Ground Truth carregado: {len(aligned_pts)} anotações no espaço alinhado.")

    # Carregar modelos
    print("[*] Carregando modelo DEF-rgbtcc (DualStreamRGBTNet)...")
    model_def = DualStreamRGBTNet()
    weight_path = ROOT_DIR / "weights" / "best_model.pth"
    if weight_path.exists():
        ckpt = torch.load(weight_path, map_location="cpu")
        model_def.load_state_dict(ckpt["model"])
        print(f"[✓] Pesos carregados de {weight_path}")
    else:
        print("[!] Checkpoint best_model.pth não encontrado.")
    model_def.to(device).eval()

    model_liu = None
    if LIUZYWEN_AVAILABLE:
        try:
            print("[*] Carregando modelo LiuzywenRGBTCCNet...")
            model_liu = build_liuzywen_model(device=device, eval_mode=True)
            print("[✓] Modelo Liuzywen pronto.")
        except Exception as e:
            print(f"[!] Erro ao instanciar Liuzywen: {e}")

    # Definir ROIs de teste com poucas pessoas
    rois_teste = [
        ("Área Vazia - Céu e Telhado", (0, 320, 0, 150)),
        ("Lateral Direita - 5 Pessoas", (832, 1152, 448, 768)),
        ("Calçada de Pedestres no Solo - 9 Pessoas", (100, 450, 700, 1000)),
        ("Canto Inferior Direito - 19 Pessoas", (768, 1280, 512, 1024)),
    ]

    resultados = []
    print("\n" + "=" * 90)
    print("           AVALIAÇÃO DE MODELOS RGBT-CC EM RECORTES COM POUCAS PESSOAS")
    print("=" * 90)
    print(f"{'Região Avaliada':40s} | {'Real':4s} | {'DEF (Int)':9s} | {'DEF (Picos)':11s} | {'Liu (Token)':11s}")
    print("-" * 90)

    for nome, roi in rois_teste:
        res = avaliar_recorte(
            img_rgb, img_th, aligned_pts, roi, nome, model_def, model_liu, device
        )
        resultados.append(res)
        real = res["pessoas_reais_ground_truth"]
        m_def = res["modelo_def_rgbtcc_estimado"]
        picos = res["modelo_def_rgbtcc_picos_locais"]
        tok = res["modelo_liuzywen_token"]
        print(f"{nome:40s} | {real:4d} | {m_def:9.1f} | {picos:11d} | {tok:11.2f}")

    print("=" * 90)

    # Exportar resultados em JSON
    json_path = OUTPUT_DIR / "relatorio_recortes_poucas_pessoas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "branch": "feat/contagem-recorte-poucas-pessoas",
            "timestamp": str(np.datetime64("now")),
            "total_cenas_avaliadas": len(resultados),
            "resultados": resultados,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Relatório detalhado salvo em: {json_path}")
    print(f"[✓] Painéis visuais salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
