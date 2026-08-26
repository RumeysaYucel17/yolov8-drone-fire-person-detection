import cv2
from ultralytics import YOLO

def run_video_demo(video_path, model_path):
    # Eğitilen en iyi modelin yüklenmesi
    model = YOLO(model_path)
    
    # Video akışının başlatılması
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[HATA] Video dosyası açılamadı: {video_path}")
        return

    print("[BİLGİ] Video işleniyor... Çıkış için 'q' tuşuna basın.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Model tahmini (Küçük nesne tespiti için conf=0.15)
        results = model.predict(source=frame, conf=0.15, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Sınıf bazlı threshold filtrelemesi
                # Fire (0): %20 üzeri, Person (1): %35 üzeri
                if (cls_id == 0 and conf >= 0.20) or (cls_id == 1 and conf >= 0.35):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[cls_id]
                    
                    # Görsel kutu çizimi
                    color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("İHA Yangın ve İnsan Tespiti Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = r"runs\detect\Deney3_YOLOv8s\weights\best.pt"
    VIDEO_PATH = "test_video.mp4"  # Test edilecek video yolu
    run_video_demo(VIDEO_PATH, MODEL_PATH)