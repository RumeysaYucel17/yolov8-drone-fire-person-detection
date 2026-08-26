# İHA/Drone Görüntülerinden Gerçek Zamanlı Yangın ve İnsan Tespiti

Bu proje, İHA (Aptal/Otonom Dronlar) tarafından çekilen aerial (kuş bakışı) video akışları üzerinden derin öğrenme yöntemleri (YOLOv8) kullanılarak **yangın (`fire`)** ve **insan (`person`)** varlığını gerçek zamanlı tespit etmek amacıyla geliştirilmiştir.

---

## İçindekiler
1. [Proje Hakkında ve Amacı](#-proje-hakkında-ve-amacı)
2. [Veri Seti Yapısı](#-veri-seti-yapısı)
3. [Gereksinimler ve Kurulum](#-gereksinimler-ve-kurulum)
4. [Klasör Hiyerarşisi](#-klasör-hiyerarşisi)
5. [Kullanım ve Çalıştırma Rehberi](#-kullanım-ve-çalıştırma-rehberi)
6. [Deneysel Sonuçlar ve Metrikler](#-deneysel-sonuçlar-ve-metrikler)
7. [Sınırlamalar ve Saha Zorlukları](#-sınırlamalar-ve-saha-zorlukları)
8. [Gelecek Çalışmalar](#-gelecek-çalışmalar)

---

## Proje Hakkında ve Amacı
Arama-kurtarma ve erken ihbar sistemlerinde İHA görüntülerinin işlenmesi kritik öneme sahiptir. Projenin temel hedefleri:
- Alev odaklarının erken safhada tespit edilerek yangın genişlemeden müdahale imkanı sağlanması.
- Afet alanlarında insan varlığının tespit edilerek arama-kurtarma ekiplerine canlı koordinat verilmesi.
- Kuş bakışı açılarda pikselleri küçülen nesnelerin (özellikle insan figurlerinin) düşük güven eşikleri ve sınıf filtreleri ile kaçırılmadan yakalanması.

---

##  Veri Seti Yapısı
Projelerdeki veri seti İHA bakış açısından çekilmiş görsellerden derlenmiştir.

- **Toplam Görsel Sayısı:** 4.572
- **Dağılım:**
  - `train/`: 4.362 görüntü (%95.4)
  - `val/`: 175 görüntü (%3.8)
  - `test/`: 35 görüntü (%0.8)
- **Sınıflar (`data.yaml`):**
  - `0`: `fire` (Yangın / Alev)
  - `1`: `person` (İnsan)

- **Dataset:** Projede kullanılan aerial/drone veri setine [ Roboflow Bağlantısı](https://universe.roboflow.com/persontrain/person-and-fire-detection) üzerinden erişebilirsiniz.
---

## Gereksinimler ve Kurulum

Projenin sorunsuz çalışması için Python 3.8+ ortamında aşağıdaki bağımlılıkların yüklenmesi gerekmektedir:

```bash
pip install ultralytics opencv-python torch torchvision matplotlib

Deneysel Sonuçlar ve Metrikler
3 farklı hipotez doğrultusunda gerçekleştirilen deneylerin sonuçları aşağıdadır:
Deney,Model,Epoch,Batch,Learning Rate,Optimizer,mAP50,mAP50-95,Açıklama / Hipotez
Deney 1,YOLOv8n,30,16,0.010,SGD,0.68,0.42,"Baseline: Yüksek FPS sağlandı, dikey küçük nesneler kaçırıldı."
Deney 2,YOLOv8s,30,16,0.010,SGD,0.76,0.51,"Model Büyütme: Katman derinliği artırıldı, mAP %8 yükseldi."
Deney 3,YOLOv8s,50,16,0.005,SGD,0.84,0.59,Optimal Model: İnce ayarlı LR ile en yüksek başarı yakalandı.

PR Eğrisi Değerleri (Deney 3):

person Sınıfı mAP50: %77.6

fire Sınıfı mAP50: %66.5

Genel Ortalamalar (All Classes): %72.0

Sınırlamalar ve Saha Zorlukları

Aşırı İrtifa ve Piksel Boyutu: Drone irtifası 50 metre üzerine çıktığında dikey duran insan figürlerinin kapladığı alan 10x10 pikselin altına düşmekte ve yalancı negatif (False Negative) oranı artmaktadır.

Optik Görüş Kısıtı ve Duman Örtüsü: Yangından yükselen yoğun duman örtüsü, alevlerin üzerini kapattığında optik (RGB) kameralar yetersiz kalmaktadır.

Arka Plan Çakışmaları: Gün ışığında parlayan çatı yüzeyleri veya sarı/kırmızı renkteki ekipmanlar zaman zaman yangın sınıfı ile karıştırılabilmektedir.

Çözüm Entegrasyonu: Bu sınırlamaları aşmak adına video_demo.py içerisinde genel güven eşiği 0.15 seviyesine çekilmiş; yangın için %20, insan için %35 dinamik sınıf eşikleri entegre edilmiştir.

Gelecek Çalışmalar

Termal (IR) Kamera Entegrasyonu: Duman arkasındaki canlıları ve ısı kaynaklarını tespit etmek için RGB ve Termal görüntülerin birleştirilmesi (Image Fusion).

Edge AI Yayınlama: Eğitilen modelin ONNX/TensorRT formatına dönüştürülerek NVIDIA Jetson Orin Nano gibi gömülü sistemlerde 60+ FPS hızda çalıştırılması.

Nesne Takibi (Object Tracking): ByteTrack algoritması eklenerek tespit edilen insanların ve yangın yayılımının anlık alan takibinin yapılması.
