import cv2
import os
import sys
import argparse

# Adiciona o diretório atual ao path para permitir imports relativos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.yolo_detector import YOLODetector
from src.reid.mobilenet_reid import MobileNetReID
from src.tracking.stable_tracker import StableTracker
from src.ui.processor import LiveProcessor


def _is_mostly_black(frame, mean_threshold=8.0, std_threshold=5.0):
    """Heurística simples para detectar feed preto/inválido."""
    mean_bgr = frame.mean(axis=(0, 1))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_gray = gray.std()
    return float(mean_bgr.mean()) < mean_threshold and float(std_gray) < std_threshold


def _build_backend_candidates(backend_mode):
    """Define ordem de tentativa dos backends de captura."""
    backend_map = {
        "msmf": [(cv2.CAP_MSMF, "MSMF")],
        "dshow": [(cv2.CAP_DSHOW, "DirectShow")],
        "auto": [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "Auto"), (cv2.CAP_DSHOW, "DirectShow")],
        "any": [(cv2.CAP_ANY, "Auto")],
    }
    return backend_map.get(backend_mode, backend_map["auto"])


def _open_camera_with_fallback(preferred_index=0, backend_mode="auto"):
    """Tenta abrir a câmera com múltiplos backends e índices no Windows."""
    backend_candidates = _build_backend_candidates(backend_mode)

    index_candidates = [preferred_index, 1, 2]
    if preferred_index != 0:
        index_candidates.append(0)

    for backend, backend_name in backend_candidates:
        for index in index_candidates:
            if backend == cv2.CAP_ANY:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok = False
            frame = None
            for _ in range(20):  # Warm-up para autoexposição/autofoco
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    break

            if not ok or frame is None or frame.size == 0:
                cap.release()
                continue

            if _is_mostly_black(frame):
                cap.release()
                continue

            return cap, index, backend_name

    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Sistema de Contagem de Pessoas em Camera ao Vivo")
    parser.add_argument("--camera", type=int, default=0, help="Indice da camera (padrao: 0)")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "msmf", "dshow", "any"],
        help="Backend da camera (padrao: auto).",
    )
    args = parser.parse_args()

    print("--- Iniciando Sistema Profissional de Contagem ---")
    
    # 1. Inicialização de Componentes
    try:
        detector = YOLODetector(model_path="yolo11n.pt", conf=0.65)
        reid = MobileNetReID()
        tracker = StableTracker(stability_frames=15)
        processor = LiveProcessor(detector, reid, tracker)
    except Exception as e:
        print(f"Erro na inicializacao: {e}")
        return

    # 2. Captura de Video
    cap, selected_index, selected_backend = _open_camera_with_fallback(args.camera, args.backend)
    if cap is None:
        print("Erro: Nao foi possivel abrir uma webcam valida (feed preto/inacessivel).")
        print("Dica: tente --camera 1 ou --backend msmf.")
        return

    print(f"Camera ativa: indice={selected_index}, backend={selected_backend}")
    print("Sistema rodando. Pressione 'q' na janela de video para sair.")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # 3. Processamento Orquestrado
            annotated_frame = processor.process_frame(frame)
            
            # 4. Exibição (OpenCV Janela Nativa para performance fora do notebook)
            cv2.imshow("Contagem Profissional (YOLOv11 + Re-ID)", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"--- Sessão Finalizada ---\nPessoas Únicas Confirmadas: {tracker.get_total_unique()}")

if __name__ == "__main__":
    main()
