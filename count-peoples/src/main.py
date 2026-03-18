import cv2
import os
import sys

# Adiciona o diretório atual ao path para permitir imports relativos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.yolo_detector import YOLODetector
from src.reid.mobilenet_reid import MobileNetReID
from src.tracking.stable_tracker import StableTracker
from src.ui.processor import LiveProcessor

def main():
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Nao foi possivel abrir a webcam.")
        return

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
