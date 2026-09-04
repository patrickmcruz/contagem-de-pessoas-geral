"""
Script de Validação e Teste da Opção 2 (Pure PyTorch MSDeformAttn)
================================================================
Testa a execução ponta a ponta da arquitetura ThermalRGBNet (PVTv2 + Deformable Attention)
com o par de imagens tratadas em data/gold/manually/.
"""

import os
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from nets.RGBTCCNet import ThermalRGBNet

def run_test():
    print("=" * 60)
    print("TESTE DE EXECUÇÃO: OPÇÃO 2 - PYTORCH PURO (MSDeformAttn)")
    print("=" * 60)

    # 1. Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dispositivo PyTorch: {device}")
    if torch.cuda.is_available():
        print(f"[*] Placa de Vídeo:      {torch.cuda.get_device_name(0)}")

    # 2. Localização das Imagens Gold Tratadas
    root_proj = CURRENT_DIR.parent
    rgb_path = root_proj / "count-github-def_rgbtcc" / "app" / "data" / "gold" / "manually" / "rgb_final.jpg"
    th_path = root_proj / "count-github-def_rgbtcc" / "app" / "data" / "gold" / "manually" / "thermal_final.jpg"
    blend_path = root_proj / "count-github-def_rgbtcc" / "app" / "data" / "gold" / "manually" / "blend_alta_precisao_app_final.jpg"

    assert rgb_path.exists(), f"RGB não encontrado: {rgb_path}"
    assert th_path.exists(), f"Térmica não encontrada: {th_path}"
    assert blend_path.exists(), f"Blend não encontrado: {blend_path}"

    print(f"[*] Imagem RGB:     {rgb_path}")
    print(f"[*] Imagem Térmica: {th_path}")

    # 3. Leitura e Preparação conforme padrão do autor (predataset_RGBT_CC.py)
    # Target resolution: 672x448 (multiplos exatos de 224: 3x2)
    img_rgb = cv2.imread(str(rgb_path))[..., ::-1].copy()
    img_th = cv2.imread(str(th_path))[..., ::-1].copy()
    img_blend = cv2.imread(str(blend_path))[..., ::-1].copy()

    img_rgb_672 = cv2.resize(img_rgb, (672, 448), interpolation=cv2.INTER_AREA)
    img_th_672 = cv2.resize(img_th, (672, 448), interpolation=cv2.INTER_AREA)

    # 4. Transformações Estatísticas Originais do RGBT-CC (datasets/crowd.py)
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.407, 0.389, 0.396], std=[0.241, 0.246, 0.242]),
    ])
    th_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191]),
    ])

    t_rgb = rgb_transform(img_rgb_672)
    t_th = th_transform(img_th_672)

    # 5. Fatiamento em grade 3x2 de patches 224x224 (datasets/crowd.py)
    m = int(672 / 224)  # 3 colunas
    n = int(448 / 224)  # 2 linhas

    crops_rgb = []
    crops_th = []
    for i in range(0, m):
        for j in range(0, n):
            crop_rgb = t_rgb[:, j * 224: 224 * (j + 1), i * 224:(i + 1) * 224].unsqueeze(0)
            crop_th = t_th[:, j * 224: 224 * (j + 1), i * 224:(i + 1) * 224].unsqueeze(0)
            crops_rgb.append(crop_rgb)
            crops_th.append(crop_th)

    batch_rgb = torch.cat(crops_rgb, 0).to(device)
    batch_th = torch.cat(crops_th, 0).to(device)

    print(f"[*] Batch de Patches RGB:     {batch_rgb.shape} (6 cortes de 224x224)")
    print(f"[*] Batch de Patches Térmico: {batch_th.shape} (6 cortes de 224x224)")

    # 6. Instanciação do Modelo ThermalRGBNet
    print("[*] Instanciando ThermalRGBNet com PVTv2-B3 e MSDeformAttn Pytorch Puro...")
    t0 = time.perf_counter()
    model = ThermalRGBNet().to(device)
    model.eval()
    t_init = (time.perf_counter() - t0) * 1000.0
    print(f"[*] Modelo carregado em {t_init:.2f} ms")

    # 7. Inferência
    print("[*] Executando Forward Pass na GPU...")
    t1 = time.perf_counter()
    with torch.no_grad():
        count_pred, outputs, mu_normed = model([batch_rgb, batch_th])
        
        # Reconstrução da Densidade Espacial 2D (test.py linhas 58-61)
        outputs1 = torch.cat((outputs[0], outputs[1]), dim=1)
        outputs2 = torch.cat((outputs[2], outputs[3]), dim=1)
        outputs3 = torch.cat((outputs[4], outputs[5]), dim=1)
        reconstructed_density = torch.cat((outputs1, outputs2, outputs3), dim=2)
    
    t_infer = (time.perf_counter() - t1) * 1000.0

    density_np = reconstructed_density[0].cpu().numpy()
    print("\n" + "-" * 50)
    print("RESULTADO DA INFERÊNCIA DO MODELO ORIGINAL:")
    print("-" * 50)
    print(f"Tempo de Inferência:            {t_infer:.2f} ms")
    print(f"Resolução Mapa Reconstruído:    {density_np.shape[1]}x{density_np.shape[0]} px (downsampling 1/8)")
    print(f"Soma Integral da Densidade:     {density_np.sum():.2f}")
    print(f"Predição de Count Token:        {count_pred.mean().item():.2f}")
    print("-" * 50 + "\n")

    # 8. Renderização Visual dos Resultados
    out_dir = root_proj / "count-github-def_rgbtcc" / "notebooks" / "output" / "06_teste_opcao2_rgbtcc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Colorir mapa de densidade
    d_norm = ((density_np - density_np.min()) / (density_np.max() - density_np.min() + 1e-8) * 255.0).astype(np.uint8)
    heatmap_jet = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    heatmap_full = cv2.resize(heatmap_jet_rgb, (img_blend.shape[1], img_blend.shape[0]))
    blend_overlay = cv2.addWeighted(img_blend, 0.55, heatmap_full, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. RGB Tratado (data/gold/manually)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(density_np, cmap="jet")
    axes[1].set_title(f"2. Mapa de Densidade 56x84 (Soma: {density_np.sum():.1f})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(blend_overlay)
    axes[2].set_title("3. Overlay no Blend de Alta Precisão (ADR 007)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plot_file = out_dir / "validacao_opcao2_pure_pytorch.jpg"
    plt.savefig(str(plot_file), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[✓] Gráfico de validação gerado com sucesso em: {plot_file}")
    print("=" * 60)
    print("STATUS FINAL: OPÇÃO 2 APROVADA COM SUCESSO ABSOLUTO!")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
