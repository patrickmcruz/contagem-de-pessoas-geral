import cv2
import sys

def check_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open {path}")
        return
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"File: {path}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    
    ret, frame = cap.read()
    if ret:
        print(f"Frame shape: {frame.shape} (H, W, C)")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_video(sys.argv[1])
    else:
        check_video(r"c:\Users\patrickcruz\Documents\Professional\Github\contagem-de-pessoas\count-peoples-record\input\exemplo_3_pessoa_input.mp4")
