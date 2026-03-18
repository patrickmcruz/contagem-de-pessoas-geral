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

def main():
    parser = argparse.ArgumentParser(description="Sistema de Contagem de Pessoas em Video Gravado")
    parser.add_argument("--input", type=str, required=True, help="Caminho para o arquivo de video de entrada")
    parser.add_argument("--output", type=str, default="output_recorded.mp4", help="Caminho para salvar o video processado")
    parser.add_argument("--show", action="store_true", help="Mostrar janela de visualizacao durante o processamento")
    args = parser.parse_args()

    print("--- Iniciando Processamento de Video Gravado ---")
    print(f"Entrada: {args.input}")
    print(f"Saida: {args.output}")
    
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
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Erro: Nao foi possivel abrir o video {args.input}")
        return

    # 3. Configuração do VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("Processando... Aguarde a finalizacao.")
    
    try:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # 4. Processamento Orquestrado
            annotated_frame = processor.process_frame(frame)
            
            # 5. Salvar Frame
            out.write(annotated_frame)
            
            # 6. Exibição Opcional
            if args.show:
                cv2.imshow("Processando Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progresso: {progress:.1f}% ({frame_count}/{total_frames})")
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"--- Processamento Finalizado ---")
        print(f"Video salvo em: {args.output}")
        print(f"Pessoas Unicas Confirmadas: {tracker.get_total_unique()}")

if __name__ == "__main__":
    main()
