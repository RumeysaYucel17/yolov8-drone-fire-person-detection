import os
import cv2
from ultralytics import YOLO

def run_camera():
    model_path = r"runs\detect\runs\detect\Deney3_YOLOv8s_LR-2\weights\best.pt"
    
    if not os.path.exists(model_path):
        model_path = r"runs\detect\Deney3_YOLOv8s_LR\weights\best.pt"

    print(f"Model yükleniyor: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[HATA] Kamera açılamadı!")
        return

    print("\n[BİLGİ] Canlı kamera testi başlatıldı! Çıkmak için 'q' tuşuna bas.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[HATA] Kameradan görüntü alınamıyor.")
            break

        # conf=0.20 ile daha hassas tespit yapıyoruz
        results = model.predict(source=frame, conf=0.55, iou=0.5, show=False)

        annotated_frame = results[0].plot()

        cv2.imshow("Drone Fire & Person Detection - Canli Kamera Testi", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[BİLGİ] Kamera testi kapatıldı.")

if __name__ == "__main__":
    run_camera()