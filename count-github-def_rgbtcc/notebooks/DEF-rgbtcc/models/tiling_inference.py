"""
Módulo de Inferência por Mosaico / Janelamento (Tiling / Sliding Patch Inference)
================================================================================
Este módulo implementa a técnica de inferência por patches deslizantes (Tiling)
para imagens aéreas de drone de altíssima resolução.

Objetivo:
Resolver o problema físico de colapso de escala (sub-pixel problem), onde cabeças
humanas medindo 3-5 pixels na imagem original desaparecem após os sucessivos
downsamplings (16x) do backbone convolucional da rede.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


def _create_blend_window(h: int, w: int) -> np.ndarray:
    """Cria uma janela 2D de ponderação suave (Hann window) para evitar artefatos de emenda."""
    wy = np.hanning(h)
    wx = np.hanning(w)
    # Evitar zeros nas bordas extremas para não dividir por zero
    wy = np.clip(wy, 0.05, 1.0)
    wx = np.clip(wx, 0.05, 1.0)
    window = np.outer(wy, wx)
    return window.astype(np.float32)


def predict_tiled_density(
    model: torch.nn.Module,
    tensor_rgb: torch.Tensor,
    tensor_th: torch.Tensor,
    patch_size: Tuple[int, int] = (512, 640),
    overlap: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Executa a inferência de densidade dividindo a imagem em patches (Tiling).

    Args:
        model: Modelo de contagem de pessoas RGBT-CC em modo eval().
        tensor_rgb: Tensor da imagem RGB [1, 3, H, W].
        tensor_th: Tensor da imagem Térmica [1, 3, H, W].
        patch_size: Tupla (patch_height, patch_width). Padrão (512, 640).
        overlap: Pixels de sobreposição entre patches adjacentes (padrão 0 para grid 2x2 exato).
        device: Dispositivo de execução (cuda/cpu).

    Returns:
        Dict contendo:
            - 'density_map': Mapa de densidade global [H, W] (numpy float32).
            - 'total_count': Soma matemática do mapa de densidade.
            - 'patch_info': Lista de metadados e contagem de cada patch.
            - 'elapsed_time': Tempo total de inferência em segundos.
            - 'fps': Taxa de quadros por segundo.
            - 'fusion_weights': Lista dos pesos adaptativos w de cada patch.
    """
    if device is None:
        device = tensor_rgb.device

    b, c, h, w = tensor_rgb.shape
    patch_h, patch_w = patch_size

    # Gerar coordenadas das janelas
    if overlap > 0:
        step_y = max(1, patch_h - overlap)
        step_x = max(1, patch_w - overlap)
        y_starts = list(range(0, h - patch_h + 1, step_y))
        if y_starts[-1] + patch_h < h:
            y_starts.append(h - patch_h)
        x_starts = list(range(0, w - patch_w + 1, step_x))
        if x_starts[-1] + patch_w < w:
            x_starts.append(w - patch_w)
    else:
        y_starts = list(range(0, h, patch_h))
        x_starts = list(range(0, w, patch_w))

    canvas = np.zeros((h, w), dtype=np.float32)
    weight_canvas = np.zeros((h, w), dtype=np.float32)

    use_blend = overlap > 0
    blend_window = _create_blend_window(patch_h, patch_w) if use_blend else np.ones((patch_h, patch_w), dtype=np.float32)

    patch_info: List[Dict[str, Any]] = []
    fusion_weights: List[float] = []

    start_time = time.time()
    model.eval()

    patch_idx = 0
    with torch.no_grad():
        for y1 in y_starts:
            y2 = min(y1 + patch_h, h)
            curr_ph = y2 - y1

            for x1 in x_starts:
                x2 = min(x1 + patch_w, w)
                curr_pw = x2 - x1

                # Extrair o recorte multimodal
                p_rgb = tensor_rgb[:, :, y1:y2, x1:x2].to(device)
                p_th = tensor_th[:, :, y1:y2, x1:x2].to(device)

                # Ajustar padding caso o patch na borda seja menor que o esperado
                pad_bottom = patch_h - curr_ph
                pad_right = patch_w - curr_pw
                if pad_bottom > 0 or pad_right > 0:
                    p_rgb = F.pad(p_rgb, (0, pad_right, 0, pad_bottom), mode="reflect")
                    p_th = F.pad(p_th, (0, pad_right, 0, pad_bottom), mode="reflect")

                # Forward pass no patch
                outputs = model(p_rgb, p_th)

                if isinstance(outputs, dict):
                    d_p = outputs["density_map"].squeeze().cpu().numpy()
                    w_val = float(outputs.get("fusion_weight", torch.tensor(0.5)).squeeze().cpu().item())
                else:
                    d_p = outputs.squeeze().cpu().numpy()
                    w_val = 0.50

                fusion_weights.append(w_val)

                # Se o modelo produzir saída em escala reduzida (ex: 1/8), interpolar para o tamanho do patch
                if d_p.shape != (patch_h, patch_w):
                    d_tensor = torch.from_numpy(d_p).unsqueeze(0).unsqueeze(0)
                    d_tensor = F.interpolate(d_tensor, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
                    # Corrigir a integral proporcionalmente ao fator de interpolação
                    scale_factor = (d_p.shape[0] * d_p.shape[1]) / (patch_h * patch_w)
                    d_p = d_tensor.squeeze().cpu().numpy() * scale_factor

                # Remover padding se tiver sido aplicado
                d_p_crop = d_p[:curr_ph, :curr_pw]
                w_crop = blend_window[:curr_ph, :curr_pw]

                # Acumular no canvas global
                canvas[y1:y2, x1:x2] += d_p_crop * w_crop
                weight_canvas[y1:y2, x1:x2] += w_crop

                patch_count = float(np.sum(d_p_crop))
                patch_max = float(np.max(d_p_crop))
                patch_name = f"Patch_{patch_idx+1}_(Y:{y1}-{y2},X:{x1}-{x2})"
                patch_info.append({
                    "id": patch_idx + 1,
                    "name": patch_name,
                    "bbox": (y1, y2, x1, x2),
                    "count": patch_count,
                    "max_val": patch_max,
                    "fusion_weight": w_val,
                })
                patch_idx += 1

    elapsed_time = time.time() - start_time
    fps = patch_idx / elapsed_time if elapsed_time > 0 else 0.0

    # Normalizar pelo peso das janelas acumuladas
    mask_weights = weight_canvas > 0
    canvas[mask_weights] /= weight_canvas[mask_weights]

    density_map = np.clip(canvas, 0, None)
    total_count = float(np.sum(density_map))

    return {
        "density_map": density_map,
        "total_count": total_count,
        "patch_info": patch_info,
        "elapsed_time": elapsed_time,
        "fps": fps,
        "num_patches": patch_idx,
        "mean_fusion_weight": float(np.mean(fusion_weights)) if fusion_weights else 0.50,
    }


def suppress_background_noise(
    density_map: np.ndarray,
    border_margin: int = 8,
    cutoff_ratio: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Aplica supressão de ruído residual de fundo e neutralização de artefatos de borda.

    Args:
        density_map: Mapa de densidade 2D [H, W] (numpy array).
        border_margin: Margem em pixels a ser zerada nas bordas da imagem para
                       eliminar artefatos de padding de convolução (padrão: 8px).
        cutoff_ratio: Fração da densidade máxima abaixo da qual os valores são
                      zerados como ruído difuso de fundo (padrão: 0.05 ou 5%).

    Returns:
        dmap_clean: Mapa de densidade limpo [H, W].
        stats: Dicionário com métricas de ruído neutralizado e percentual de área zerada.
    """
    dmap_clean = density_map.copy().astype(np.float32)
    raw_sum = float(np.sum(dmap_clean))
    d_max = float(np.max(dmap_clean))

    # 1. Neutralizar reflexões de borda de convolução
    if border_margin > 0:
        h, w = dmap_clean.shape
        dmap_clean[:border_margin, :] = 0.0
        dmap_clean[h - border_margin:, :] = 0.0
        dmap_clean[:, :border_margin] = 0.0
        dmap_clean[:, w - border_margin:] = 0.0

    # 2. Limiarização do piso de ruído difuso em áreas vazias (asfalto, grama)
    if cutoff_ratio > 0.0 and d_max > 0.0:
        cutoff = cutoff_ratio * d_max
        dmap_clean[dmap_clean < cutoff] = 0.0

    clean_sum = float(np.sum(dmap_clean))
    suppressed_count = raw_sum - clean_sum
    zero_ratio = float(np.mean(dmap_clean == 0.0) * 100.0)

    stats = {
        "raw_sum": raw_sum,
        "clean_sum": clean_sum,
        "suppressed_count": suppressed_count,
        "zero_area_percent": zero_ratio,
        "cutoff_threshold": cutoff_ratio * d_max if d_max > 0 else 0.0,
    }
    return dmap_clean, stats

