import os
from ultralytics import YOLO

def run_demo():

    model_path = r"runs\detect\runs\detect\Deney3_YOLOv8s_LR-2\weights\best.pt"
    
    if not os.path.exists(model_path):
        model_path = r"runs\detect\Deney3_YOLOv8s_LR\weights\best.pt"

    print(f"Model yükleniyor: {model_path}")
    model = YOLO(model_path)
    
    test_images_path = r"dataset\test\images"
    
    print("Test görselleri üzerinde çıkarım (inference) yapılıyor...")
    results = model.predict(
        source=test_images_path,
        conf=0.35,
        save=True,
        project="runs/detect",
        name="Test_Sonuclari"
    )
    
    print("\n[BAŞARILI] Test sonuçları 'runs/detect/Test_Sonuclari' klasörüne kaydedildi!")

if __name__ == "__main__":
    run_demo()