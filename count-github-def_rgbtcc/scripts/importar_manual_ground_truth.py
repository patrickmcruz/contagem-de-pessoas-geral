#!/usr/bin/env python3
"""
Importador e Estruturador de Contagem Manual (Ground Truth)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Origem dos dados:
    data/manual_counting_check/
    Neste diretório temos as imagens e os respectivos arquivos .txt correspondentes
    com a localização espacial das pessoas/cabeças ('head/people' location),
    identificadas por pares espaciais (X, Y).

Destino estruturado:
    data/ground_truth/{STEM}/
    Gera a estrutura contratual padronizada consumida pelo pipeline de inferência,
    mapas de densidade e telemetria:
    - pontos_ground_truth_{STEM}.csv e pontos_ground_truth.csv
    - pontos_ground_truth_{STEM}.json e pontos_ground_truth.json
    - checkpoint_{STEM}.json e checkpoint_anotacao.json
    - ground_truth_aligned_1280x1024_{STEM}.json e ground_truth_aligned_1280x1024.json
    - rgb_anotada_ground_truth_{STEM}.jpg
    - metadados_{STEM}.json e metadados.json
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pandas as pd
except ImportError:
    pd = None

import csv
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "manual_counting_check"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ground_truth"


def carregar_coordenadas_txt(caminho_txt: Path) -> List[Tuple[float, float]]:
    """
    Lê o arquivo .txt contendo pares de coordenadas (X, Y) separados por espaço.
    """
    pontos = []
    with open(caminho_txt, "r", encoding="utf-8") as f:
        for idx, linha in enumerate(f, start=1):
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            partes = linha.split()
            if len(partes) >= 2:
                try:
                    x = float(partes[0])
                    y = float(partes[1])
                    pontos.append((x, y))
                except ValueError:
                    print(f"  [!] Aviso: Linha {idx} em {caminho_txt.name} inválida: '{linha}'")
    return pontos


def processar_par(
    caminho_img: Path,
    caminho_txt: Path,
    diretorio_destino_base: Path,
    largura_alinhada: int = 1280,
    altura_alinhada: int = 1024,
) -> Dict[str, Any]:
    """
    Processa um par imagem-txt e gera todos os artefatos de Ground Truth.
    """
    stem = caminho_img.stem
    pasta_gt = diretorio_destino_base / stem
    pasta_gt.mkdir(parents=True, exist_ok=True)

    # 1. Carregar imagem para validar dimensões
    with Image.open(caminho_img) as pil_img:
        w_orig, h_orig = pil_img.size
        canais = len(pil_img.getbands())

    # 2. Carregar pontos
    pontos_raw = carregar_coordenadas_txt(caminho_txt)
    total_pessoas = len(pontos_raw)

    print(f"\n[*] Processando {stem}:")
    print(f"    ├─ Imagem: {caminho_img.name} ({w_orig}x{h_orig} px, {canais} canais)")
    print(f"    ├─ Anotações: {caminho_txt.name} ({total_pessoas} cabeças identificadas)")
    print(f"    └─ Destino: {pasta_gt}")

    # Estruturar lista de coordenadas RAW com ID único
    coordenadas = []
    for idx, (x, y) in enumerate(pontos_raw, start=1):
        # Arredondar para 2 casas decimais e manter consistência
        coordenadas.append({
            "id": idx,
            "x": round(x, 2),
            "y": round(y, 2),
        })

    # 3. Gerar e salvar CSVs (com e sem stem)
    p_csv_stem = pasta_gt / f"pontos_ground_truth_{stem}.csv"
    p_csv_canon = pasta_gt / "pontos_ground_truth.csv"
    if pd is not None:
        df = pd.DataFrame(coordenadas)
        df.to_csv(p_csv_stem, index=False)
    else:
        with open(p_csv_stem, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["id", "x", "y"])
            writer.writeheader()
            writer.writerows(coordenadas)
    shutil.copyfile(p_csv_stem, p_csv_canon)

    # 4. Gerar e salvar JSONs de pontos RAW
    data_finalizacao = time.strftime("%Y-%m-%d %H:%M:%S")
    telemetria_gt = {
        "status": "finalizado",
        "data_finalizacao": data_finalizacao,
        "imagem_origem": str(caminho_img.relative_to(PROJECT_ROOT) if caminho_img.is_relative_to(PROJECT_ROOT) else caminho_img),
        "arquivo_nome": caminho_img.name,
        "resolucao_original": {
            "largura": int(w_orig),
            "altura": int(h_orig),
            "canais": int(canais),
        },
        "total_pessoas_anotadas": total_pessoas,
        "pontos": coordenadas,
    }
    p_json_stem = pasta_gt / f"pontos_ground_truth_{stem}.json"
    p_json_canon = pasta_gt / "pontos_ground_truth.json"
    with open(p_json_stem, "w", encoding="utf-8") as f:
        json.dump(telemetria_gt, f, indent=2, ensure_ascii=False)
    shutil.copyfile(p_json_stem, p_json_canon)

    # 5. Gerar e salvar Checkpoint
    checkpoint_data = {
        "tipo": "checkpoint_progresso",
        "data_checkpoint": data_finalizacao,
        "imagem_origem": str(caminho_img.relative_to(PROJECT_ROOT) if caminho_img.is_relative_to(PROJECT_ROOT) else caminho_img),
        "arquivo_nome": caminho_img.name,
        "resolucao_original": {
            "largura": int(w_orig),
            "altura": int(h_orig),
        },
        "total_pessoas_anotadas": total_pessoas,
        "pontos": coordenadas,
    }
    p_chk_stem = pasta_gt / f"checkpoint_{stem}.json"
    p_chk_canon = pasta_gt / "checkpoint_anotacao.json"
    with open(p_chk_stem, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    shutil.copyfile(p_chk_stem, p_chk_canon)

    # 6. Gerar coordenadas alinhadas no espaço de inferência (1280x1024)
    scale_x = float(largura_alinhada) / float(w_orig)
    scale_y = float(altura_alinhada) / float(h_orig)

    pontos_aligned = []
    for pt in coordenadas:
        px = int(round(pt["x"] * scale_x))
        py = int(round(pt["y"] * scale_y))
        # Garantir clamp dentro do grid de inferência
        px = max(0, min(largura_alinhada - 1, px))
        py = max(0, min(altura_alinhada - 1, py))
        pontos_aligned.append({
            "id": pt["id"],
            "x": px,
            "y": py,
        })

    gt_aligned_data = {
        "cena": stem,
        "arquivo_origem": caminho_img.name,
        "resolucao_alinhada": {
            "largura": largura_alinhada,
            "altura": altura_alinhada,
        },
        "total_pessoas_anotadas": len(pontos_aligned),
        "pontos": pontos_aligned,
    }
    p_al_stem = pasta_gt / f"ground_truth_aligned_{largura_alinhada}x{altura_alinhada}_{stem}.json"
    p_al_canon = pasta_gt / f"ground_truth_aligned_{largura_alinhada}x{altura_alinhada}.json"
    with open(p_al_stem, "w", encoding="utf-8") as f:
        json.dump(gt_aligned_data, f, indent=2, ensure_ascii=False)
    shutil.copyfile(p_al_stem, p_al_canon)

    # 7. Renderizar imagem com pontos para auditoria visual
    p_img_audit = pasta_gt / f"rgb_anotada_ground_truth_{stem}.jpg"
    raio = max(2, int(round(w_orig / 600)))
    if cv2 is not None:
        img_bgr = cv2.imread(str(caminho_img))
        if img_bgr is not None:
            img_audit = img_bgr.copy()
            for pt in coordenadas:
                cx = int(round(pt["x"]))
                cy = int(round(pt["y"]))
                cv2.circle(img_audit, (cx, cy), raio, (0, 0, 255), -1)      # Centro Vermelho
                cv2.circle(img_audit, (cx, cy), raio + 1, (0, 255, 255), 1)  # Borda Amarela
            cv2.imwrite(str(p_img_audit), img_audit)
        else:
            p_img_audit = None
    else:
        try:
            with Image.open(caminho_img) as pil_orig:
                pil_audit = pil_orig.convert("RGB")
                draw = ImageDraw.Draw(pil_audit)
                for pt in coordenadas:
                    cx = int(round(pt["x"]))
                    cy = int(round(pt["y"]))
                    draw.ellipse([cx - raio, cy - raio, cx + raio, cy + raio], fill=(255, 0, 0), outline=(255, 255, 0))
                pil_audit.save(p_img_audit, quality=92)
        except Exception as e:
            print(f"  [!] Erro ao renderizar imagem anotada com PIL: {e}")
            p_img_audit = None

    # 8. Gerar metadados da cena
    meta_info = {
        "imagem_contada": stem,
        "imagem_rgb": caminho_img.name,
        "status_anotacao": "finalizado",
        "data_finalizacao": data_finalizacao,
        "total_pessoas_anotadas": total_pessoas,
        "resolucao_raw": {
            "rgb": [int(w_orig), int(h_orig)],
        },
        "resolucao_alinhada_inferencia": [largura_alinhada, altura_alinhada],
        "arquivos": {
            "checkpoint": p_chk_stem.name,
            "pontos_raw_json": p_json_stem.name,
            "pontos_raw_csv": p_csv_stem.name,
            "pontos_alinhados_json": p_al_stem.name,
            "auditoria_visual_jpg": p_img_audit.name if p_img_audit else None,
        },
    }
    p_meta_stem = pasta_gt / f"metadados_{stem}.json"
    p_meta_canon = pasta_gt / "metadados.json"
    with open(p_meta_stem, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    shutil.copyfile(p_meta_stem, p_meta_canon)

    return {
        "stem": stem,
        "total": total_pessoas,
        "resolucao_raw": (w_orig, h_orig),
        "resolucao_alinhada": (largura_alinhada, altura_alinhada),
        "pasta": pasta_gt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Importa contagens manuais (pares imagem e .txt) para a estrutura padrão data/ground_truth/"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Diretório com os pares imagem e txt (padrão: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório raiz de Ground Truth (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--largura-alinhada",
        type=int,
        default=1280,
        help="Largura do espaço de inferência (padrão: 1280)",
    )
    parser.add_argument(
        "--altura-alinhada",
        type=int,
        default=1024,
        help="Altura do espaço de inferência (padrão: 1024)",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("=" * 76)
    print("      IMPORTAÇÃO DE GROUND TRUTH: MANUAL COUNTING -> GROUND TRUTH")
    print("=" * 76)
    print(f"Diretório de Entrada: {input_dir}")
    print(f"Diretório de Saída:   {output_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Localizar imagens
    extensoes_imagem = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    arquivos_img = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix in extensoes_imagem
    ])

    if not arquivos_img:
        print(f"[!] Nenhuma imagem encontrada em {input_dir}")
        return

    resultados = []
    for caminho_img in arquivos_img:
        stem = caminho_img.stem
        caminho_txt = input_dir / f"{stem}.txt"
        if not caminho_txt.exists():
            print(f"[!] Arquivo de anotações não encontrado para {caminho_img.name} (esperado: {caminho_txt.name}). Ignorando.")
            continue

        res = processar_par(
            caminho_img=caminho_img,
            caminho_txt=caminho_txt,
            diretorio_destino_base=output_dir,
            largura_alinhada=args.largura_alinhada,
            altura_alinhada=args.altura_alinhada,
        )
        resultados.append(res)

    print("\n" + "=" * 76)
    print("                       RESUMO DA IMPORTAÇÃO")
    print("=" * 76)
    total_geral = sum(r["total"] for r in resultados)
    for r in resultados:
        print(f"  [✓] {r['stem']:<15} | RAW: {r['resolucao_raw'][0]}x{r['resolucao_raw'][1]} | ALIGN: {r['resolucao_alinhada'][0]}x{r['resolucao_alinhada'][1]} | Total: {r['total']:>5} pessoas")
    print("-" * 76)
    print(f"  TOTAL DE CENAS IMPORTADAS: {len(resultados)}")
    print(f"  TOTAL DE INDIVÍDUOS:       {total_geral} pessoas reais anotadas")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
