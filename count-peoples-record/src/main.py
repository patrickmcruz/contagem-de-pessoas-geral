import cv2
import os
import sys
import argparse
import yaml
import logging
import time

# Adiciona o diretório atual ao path para permitir imports relativos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.yolo_detector import YOLODetector
from src.reid.mobilenet_reid import MobileNetReID
from src.tracking.stable_tracker import StableTracker
from src.ui.processor import LiveProcessor
from src.core.logger_setup import setup_logger

def load_config(config_path="config.yaml"):
    """Carrega as configuracoes de um arquivo YAML."""
    if not os.path.exists(config_path):
        # Validação básica - se o arquivo nao existe, podemos criar/usar defaults se desejado
        # Mas para este caso, o usuário solicitou o arquivo.
        print(f"Erro: Arquivo de configuracao {config_path} nao encontrado!")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time() # Inicia cronometro
    
    parser = argparse.ArgumentParser(description="Sistema de Contagem de Pessoas em Video Gravado")
    parser.add_argument("--input", type=str, required=True, help="Caminho para o arquivo de video de entrada")
    parser.add_argument("--output", type=str, default="output_recorded.mp4", help="Caminho para salvar o video processado")
    parser.add_argument("--show", action="store_true", help="Mostrar janela de visualizacao durante o processamento")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para o arquivo de configuracao")
    args = parser.parse_args()

    # 1. Carregar Configs e Setup Logger
    config = load_config(args.config)
    logger = setup_logger(
        log_level=config['logging']['level'],
        log_dir=config['logging']['log_dir'],
        filename=config['logging']['filename']
    )

    logger.info("--- Iniciando Processamento de Video Gravado ---")
    logger.info(f"Entrada: {args.input}")
    logger.info(f"Saida: {args.output}")
    
    # 2. Inicialização de Componentes baseado em config.yaml
    try:
        cfg_app = config['application']
        cfg_detect = config['detection']
        cfg_track = config['tracking']
        cfg_output = config.get('output', {}) # Fallback se nao existir
        
        device = cfg_app.get('device', 'cuda')
        use_half = cfg_app.get('use_half_precision', True)
        
        logger.debug(f"Carregando detector {cfg_detect['model_path']} (sz={cfg_detect['imgsz']}) no dispositivo {device}")
        
        detector = YOLODetector(
            model_path=cfg_detect['model_path'], 
            conf=cfg_detect['confidence'], 
            imgsz=cfg_detect['imgsz'],
            classes=cfg_detect.get('classes', [0]),
            device=device,
            use_half=use_half
        )
        
        reid = MobileNetReID(
            device=device,
            use_half=use_half
        )
        
        tracker = StableTracker(
            reid_threshold=cfg_track['reid_threshold'], 
            merge_threshold=cfg_track['merge_threshold'], 
            stability_frames=cfg_track['stability_frames']
        )
        
        processor = LiveProcessor(
            detector=detector, 
            reid=reid, 
            tracker=tracker, 
            blur_threshold=cfg_track['blur_threshold'],
            enable_counting=cfg_app['enable_counting'],
            show_status_bar=cfg_app['show_status_bar']
        )
        
    except Exception as e:
        logger.error(f"Erro fatal na inicializacao dos componentes: {str(e)}", exc_info=True)
        return

    # 3. Captura de Video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Erro: Nao foi possivel abrir o video {args.input}")
        return

    # Ativar rotação automática
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    # Obter dimensões reais
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Propriedades do vídeo carregado: {width}x{height} @ {fps:.2f} FPS")
    
    # 4. Configurar VideoWriter com parametros do YAML
    output_filename = args.output if args.output != "output_recorded.mp4" else cfg_output.get('default_filename', 'output.mp4')
    fourcc_str = cfg_output.get('fourcc', 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    logger.info("Processamento iniciado. Aguarde a finalizacao...")
    
    try:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Processamento
            annotated_frame = processor.process_frame(frame)
            
            # Salvar Frame
            out.write(annotated_frame)
            
            # Exibição Opcional
            if args.show:
                cv2.imshow("Processando Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.warning("Processamento interrompido pela tecla 'q'.")
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Progresso: {progress:.1f}% ({frame_count}/{total_frames})")
                
    except KeyboardInterrupt:
        logger.warning("\nProcessamento interrompido pelo usuario (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"Erro durante o processamento de frames: {str(e)}", exc_info=True)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"--- Processamento Finalizado ---")
        logger.info(f"Tempo Total de Execucao: {duration:.2f} segundos")
        logger.info(f"Video salvo em: {args.output}")
        if config['application']['enable_counting']:
            logger.info(f"Pessoas Unicas Confirmadas: {tracker.get_total_unique()}")
        else:
            logger.info("Contagem de pessoas estava desativada.")

if __name__ == "__main__":
    main()
