"""
Script de Inferência Direta com as Imagens Originais de notebooks/test_01/
========================================================================
Utiliza o par multimodal (RGB e Térmica) individualmente, SEM usar qualquer
imagem de fusão/overlay/blend, projetando o mapa de densidade diretamente sobre
a imagem óptica original e sobre a imagem térmica original.
"""

import os
import sys
import time
import json
from pathlib import Path

# Adicionar caminhos para importação dos módulos
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RGBTCC_DIR = ROOT_DIR.parent / "liuzywen-RGBTCC"

for p in [str(CURRENT_DIR), str(ROOT_DIR), str(RGBTCC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from nets.RGBTCCNet import ThermalRGBNet

def main():
    print("=" * 65)
    print("INFERÊNCIA COM IMAGENS ORIGINAIS DE notebooks/test_01/ (SEM OVERLAY)")
    print("=" * 65)

    # 1. Caminhos das imagens originais individuais
    p_rgb = ROOT_DIR / "notebooks" / "test_01" / "02_rgb_equalized_undistort_shift.jpg"
    p_th = ROOT_DIR / "notebooks" / "test_01" / "03_thermal_equalized_clahe.jpg"
    out_dir = ROOT_DIR / "notebooks" / "output" / "07_test_01_sem_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert p_rgb.exists(), f"Arquivo RGB não encontrado: {p_rgb}"
    assert p_th.exists(), f"Arquivo Térmico não encontrado: {p_th}"

    print(f"[*] Imagem RGB:     {p_rgb.name}")
    print(f"[*] Imagem Térmica: {p_th.name}")

    # 2. Carregamento com OpenCV
    img_rgb_bgr = cv2.imread(str(p_rgb))
    img_th_bgr = cv2.imread(str(p_th))

    img_rgb = cv2.cvtColor(img_rgb_bgr, cv2.COLOR_BGR2RGB)
    img_th = cv2.cvtColor(img_th_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f"[*] Resolução Original: {orig_w}x{orig_h} px")

    # 3. Pré-processamento: Redimensionamento para grade de 224 (672x448 = 3 colunas x 2 linhas)
    target_w, target_h = 672, 448
    rgb_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    th_resized = cv2.resize(img_th, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 4. Normalização estatística do RGBT-CC
    norm_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.407, 0.389, 0.396], std=[0.241, 0.246, 0.242])
    ])
    norm_th = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191])
    ])

    t_rgb = norm_rgb(rgb_resized)
    t_th = norm_th(th_resized)

    # 5. Fatiamento em 6 cortes de 224x224 px
    crops_rgb, crops_th = [], []
    for i in range(3):
        for j in range(2):
            c_rgb = t_rgb[:, j*224:(j+1)*224, i*224:(i+1)*224].unsqueeze(0)
            c_th = t_th[:, j*224:(j+1)*224, i*224:(i+1)*224].unsqueeze(0)
            crops_rgb.append(c_rgb)
            crops_th.append(c_th)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_rgb = torch.cat(crops_rgb, dim=0).to(device)
    batch_th = torch.cat(crops_th, dim=0).to(device)
    print(f"[*] Lote de tensores para GPU: {batch_rgb.shape} no dispositivo {device}")

    # 6. Carregamento do modelo e execução do forward pass
    print("[*] Instanciando ThermalRGBNet (Opção 2 - Pure PyTorch Deformable Attention)...")
    model = ThermalRGBNet().to(device)
    model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        count_pred, outputs, mu_normed = model([batch_rgb, batch_th])
        # Reconstrução da grade contínua 2D
        outputs1 = torch.cat((outputs[0], outputs[1]), dim=1)
        outputs2 = torch.cat((outputs[2], outputs[3]), dim=1)
        outputs3 = torch.cat((outputs[4], outputs[5]), dim=1)
        density_map = torch.cat((outputs1, outputs2, outputs3), dim=2)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    density_np = density_map[0].cpu().numpy()
    total_count = float(density_np.sum())

    print("\n" + "-" * 50)
    print("RELATÓRIO DE INFERÊNCIA COM IMAGENS ORIGINAIS:")
    print("-" * 50)
    print(f"Tempo de Execução (GPU):        {latency_ms:.2f} ms")
    print(f"Resolução do Mapa de Densidade: {density_np.shape[1]}x{density_np.shape[0]} px")
    print(f"Soma Integral da Densidade:     {total_count:.2f}")
    print("-" * 50 + "\n")

    # 7. Geração do Heatmap colorido e projeções diretas
    d_norm = ((density_np - density_np.min()) / (density_np.max() - density_np.min() + 1e-8) * 255.0).astype(np.uint8)
    heatmap_jet = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    heatmap_full_res = cv2.resize(heatmap_jet_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Projeção DIRETA sobre o RGB original (SEM USAR IMAGEM DE OVERLAY!)
    overlay_on_rgb = cv2.addWeighted(img_rgb, 0.60, heatmap_full_res, 0.40, 0)

    # Projeção DIRETA sobre o Térmico original (SEM USAR IMAGEM DE OVERLAY!)
    overlay_on_th = cv2.addWeighted(img_th, 0.60, heatmap_full_res, 0.40, 0)

    # Gravar imagens no disco
    f_rgb_overlay = out_dir / "heatmap_sobre_rgb.jpg"
    f_th_overlay = out_dir / "heatmap_sobre_termica.jpg"
    f_raw_heatmap = out_dir / "heatmap_puro_densidade.jpg"
    f_panel = out_dir / "painel_test_01_rgb_termica.jpg"

    cv2.imwrite(str(f_rgb_overlay), cv2.cvtColor(overlay_on_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(f_th_overlay), cv2.cvtColor(overlay_on_th, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(f_raw_heatmap), heatmap_jet)

    # 8. Renderizar painel consolidado 2x2
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Imagem RGB Original (notebooks/test_01/)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_th)
    axes[0, 1].set_title("2. Imagem Térmica Original (notebooks/test_01/)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    im_d = axes[1, 0].imshow(density_np, cmap="jet")
    axes[1, 0].set_title(f"3. Mapa de Densidade Puro ({density_np.shape[1]}x{density_np.shape[0]} px)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im_d, ax=axes[1, 0], fraction=0.035, pad=0.04)

    axes[1, 1].imshow(overlay_on_rgb)
    axes[1, 1].set_title("4. Heatmap Projetado Diretamente sobre o RGB Original", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(str(f_panel), dpi=150, bbox_inches="tight")
    plt.close()

    print("[✓] Processamento concluído com sucesso!")
    print(f"[*] Painel 2x2 salvo em:          {f_panel}")
    print(f"[*] Heatmap sobre RGB salvo em:     {f_rgb_overlay}")
    print(f"[*] Heatmap sobre Térmica salvo em: {f_th_overlay}")
    print("=" * 65)

if __name__ == "__main__":
    main()
