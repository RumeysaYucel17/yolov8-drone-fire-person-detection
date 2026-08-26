import os
from ultralytics import YOLO

# Veri seti yapılandırma dosyası yolu
DATA_YAML = r"dataset\data.yaml"

def run_deney_1():
    """
    DENEY 1: Baseline (Referans) Testi
    - Model: YOLOv8n (Nano - En hafif ve hızlı model)
    - Epoch: 30
    - Learning Rate: 0.01 (Varsayılan)
    - Amaç: Başlangıç başarımını ve hızı gözlemlemek.
    """
    print("\n--- DENEY 1 BAŞLATILIYOR (YOLOv8n Baseline) ---")
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.01,
        optimizer="SGD",
        name="Deney1_YOLOv8n",
        device=0,
        exist_ok=True
    )

def run_deney_2():
    """
    DENEY 2: Model Karmaşıklığını Artırma
    - Model: YOLOv8s (Small - Daha derin mimari)
    - Epoch: 30
    - Learning Rate: 0.01 (Varsayılan)
    - Amaç: Model kapasitesini artırarak küçük nesne tespit başarısını yükseltmek.
    """
    print("\n--- DENEY 2 BAŞLATILIYOR (YOLOv8s Model Büyütme) ---")
    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.01,
        optimizer="SGD",
        name="Deney2_YOLOv8s",
        device=0,
        exist_ok=True
    )

def run_deney_3():
    """
    DENEY 3: Hiperparametre İnce Ayarı (En İyi Model)
    - Model: YOLOv8s (Small)
    - Epoch: 50 (Daha uzun eğitim süresi)
    - Learning Rate: 0.005 (Daha hassas ağırlık güncellemeleri)
    - Amaç: İnce ayar yaparak en yüksek mAP50 başarımına ulaşmak.
    """
    print("\n--- DENEY 3 BAŞLATILIYOR (YOLOv8s Fine-Tuning - En İyi Model) ---")
    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.005,
        lrf=0.01,
        optimizer="SGD",
        name="Deney3_YOLOv8s",
        device=0,
        exist_ok=True
    )

if __name__ == "__main__":
    print("Çalıştırmak istediğiniz eğitimi alt k kısımdan seçebilirsiniz:")
    
    # Varsayılan olarak en başarılı deneyimiz olan Deney 3 çalışır:
    run_deney_3()
    
    # İstenildiği takdirde diğer deneyler de sırayla çalıştırılabilir:
    # run_deney_1()
    # run_deney_2()