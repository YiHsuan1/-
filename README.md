# Efficient Object Detection with YOLOV8 and KerasCV（使用YOLOV8和KerasCV進行高效能物件偵測）
南華大學跨領域-人工智慧期中報告 11123024楊佳宜 11123007陳奕瑄 11118128 吳佳恩

介紹
KerasCV 是 Keras 用於電腦視覺任務的擴展。在此範例中，我們將了解如何使用 KerasCV 訓練 YOLOV8 目標偵測模型。
KerasCV 包含針對流行電腦視覺資料集的預訓練模型，例如 ImageNet、COCO 和 Pascal VOC，可用於遷移學習。KerasCV 還提供了一系列視覺化工具，用於檢查模型學習的中間表示以及可視化物件偵測和分割任務的結果。
如果您有興趣了解使用 KerasCV 進行物件偵測，我強烈建議您查看 lukewood 建立的指南。此資源可在 使用 KerasCV 進行物件偵測中取得，全面概述了使用 KerasCV 建構物件偵測模型所需的基本概念和技術。
```python
!pip install —upgrade git+https://github.com/keras-team/keras-cv -q
```
設定
```python
import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
```
載入數據

在本指南中，我們將使用從 roboflow獲得的自動駕駛汽車資料集。為了使資料集更易於管理，我提取了較大資料集的子集，該資料集最初由 15,000 個資料樣本組成。從這個子集中，我選擇了 7,316 個樣本進行模型訓練。
為了簡化手頭上的任務並集中精力，我們將使用更少數量的物件類別。具體來說，我們將考慮五個主要類別進行檢測和分類：汽車、行人、交通燈、騎自行車的人和卡車。這些類別代表了自動駕駛汽車中遇到的一些最常見和最重要的物件。
透過將資料集縮小到這些特定類別，我們可以集中精力建立強大的物件偵測模型，該模型可以準確地識別和分類這些重要物件。
TensorFlow Datasets 庫提供了一種下載和使用各種資料集（包括物件偵測資料集）的便捷方法。對於想要快速開始處理資料而無需手動下載和預處理資料的人來說，這可能是一個不錯的選擇。
您可以在此處查看各種物件偵測資料集 TensorFlow 資料集
但是，在此程式碼範例中，我們將示範如何使用 TensorFlow 的管道從頭開始載入資料集tf.data。這種方法提供了更大的靈活性，並允許您根據需要自訂預處理步驟。
載入 TensorFlow 資料集庫中不可用的自訂資料集是使用tf.data管道的主要優點之一。此方法可讓您建立適合資料集的特定需求和要求的自訂資料預處理管道。

超參數
```python
SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0
```
建立字典以將每個類別名稱對應到唯一的數字識別碼。此映射用於在物件偵測任務的訓練和推理過程中對類別標籤進行編碼和解碼。
```python
class_ids = [
    "car",
    "pedestrian",
    "trafficLight",
    "biker",
    "truck",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Path to images and annotations
path_images = "/kaggle/input/dataset/data/images/"
path_annot = "/kaggle/input/dataset/data/annotations/"

# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)

# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
        if file_name.endswith(".jpg")
    ]
)
```
下面的函數讀取 XML 檔案並尋找影像名稱和路徑，然後迭代 XML 檔案中的每個物件以提取每個物件的邊界框座標和類別標籤。

此函數傳回三個值：影像路徑、邊界框清單（每個邊界框表示為四個浮點數的清單：xmin、ymin、xmax、ymax）以及與每個邊界框對應的類別ID 清單（表示為整數） 。類別 ID 是透過使用名為 的字典將類別標籤對應到整數值來獲得的class_mapping。
```python
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)
```
在這裡，我們使用和 列表tf.ragged.constant創建不規則張量。參差不齊的張量是一種可以沿著一個或多個維度處理不同長度的資料的張量。這在處理具有可變長度序列的資料（例如文字或時間序列資料）時非常有用。bboxclasses
```python
classes = [
    [8, 8, 8, 8, 8],      # 5 classes
    [12, 14, 14, 14],     # 4 classes
    [1],                  # 1 class
    [7, 7],               # 2 classes
 ...]
bbox = [
    [[199.0, 19.0, 390.0, 401.0],
    [217.0, 15.0, 270.0, 157.0],
    [393.0, 18.0, 432.0, 162.0],
    [1.0, 15.0, 226.0, 276.0],
    [19.0, 95.0, 458.0, 443.0]],     #image 1 has 4 objects
    [[52.0, 117.0, 109.0, 177.0]],   #image 2 has 1 object
    [[88.0, 87.0, 235.0, 322.0],
    [113.0, 117.0, 218.0, 471.0]],   #image 3 has 2 objects
 ...]
```
在這種情況下，每個影像的bbox和classes列表具有不同的長度，具體取決於影像中的物件數量以及相應的邊界框和類別。為了處理這種變化，使用不規則張量而不是常規張量。
後來，這些不規則的張量被用來創造一個tf.data.Dataset使用 from_tensor_slices方法。此方法透過沿著第一維度對輸入張量進行切片來建立資料集。透過使用參差不齊的張量，數據集可以處理每個影像的不同長度的數據，並為進一步處理提供靈活的輸入管道。
```python
bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
```
將數據拆分為訓練數據和驗證數據
```python
# Determine the number of validation samples
num_val = int(len(xml_files) * SPLIT_RATIO)

# Split the dataset into train and validation sets
val_data = data.take(num_val)
train_data = data.skip(num_val)
```
讓我們看看資料載入和邊界框格式以使事情順利進行。KerasCV 中的邊界框具有預定的格式。為此，您必須將邊界框捆綁到符合下列要求的字典中：
```python
bounding_boxes = {
    # num_boxes may be a Ragged dimension
    'boxes': Tensor(shape=[batch, num_boxes, 4]),
    'classes': Tensor(shape=[batch, num_boxes])
}
```
該字典有兩個鍵'boxes'和'classes'，每個鍵都對應到 TensorFlow RaggedTensor 或 Tensor 物件。張'boxes'量的形狀為[batch, num_boxes, 4]，其中batch是批次中圖像的數量，num_boxes是任何圖像中邊界框的最大數量。4 表示定義邊界框所需的四個值：xmin、ymin、xmax、ymax。

張'classes'量的形狀為[batch, num_boxes]，其中每個元素代表張量中對應邊界框的類別標籤'boxes'。num_boxes 維度可能參差不齊，這意味著批次中的圖像之間的框數量可能會有所不同。

最終的字典應該是：
```python
{"images": images, "bounding_boxes": bounding_boxes}
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}
```
在這裡，我們建立一個圖層，將影像大小調整為 640x640 像素，同時保持原始縱橫比。與影像關聯的邊界框在 xyxy格式中指定。如有必要，調整大小的影像將用零填充以保持原始縱橫比。

KerasCV 支援的邊界框格式： 1. CENTER_XYWH 2. XYWH 3. XYXY 4. REL_XYXY 5. REL_XYWH 6. YXYX 7. REL_YXYX

您可以在文件中閱讀有關 KerasCV 邊界框格式的更多資訊 。

此外，可以在任兩對之間執行格式轉換：
```python
boxes = keras_cv.bounding_box.convert_format(
        bounding_box,
        images=image,
        source="xyxy",  # Original Format
        target="xywh",  # Target Format (to which we want to convert)
    )
```
數據增強
建構物件檢測管道時最具挑戰性的任務之一是資料增強。它涉及對輸入圖像應用各種變換，以增加訓練資料的多樣性並提高模型的泛化能力。然而，在處理物件偵測任務時，它變得更加複雜，因為這些轉換需要了解底層邊界框並相應地更新它們。

KerasCV 為邊界框增強提供本機支援。KerasCV 提供了大量專門用於處理邊界框的資料增強層。這些圖層在影像轉換時智慧地調整邊界框座標，確保邊界框保持準確並與增強影像對齊。

透過利用 KerasCV 的功能，開發人員可以方便地將邊界框友好的資料增強整合到他們的物件偵測管道中。透過在 tf.data 管道中執行即時增強，該過程變得無縫且高效，從而實現更好的訓練和更準確的物件偵測結果。
```python
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
        keras_cv.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
        ),
    ]
)
```
建立訓練資料集
```python
train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
```
視覺化
```python
def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)

visualize_dataset(
    val_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)
```
我們需要從預處理字典中提取輸入並準備好將其輸入模型。
```python
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
```
創建模型
YOLOv8 是一種前沿的 YOLO 模型，可用於各種電腦視覺任務，例如物件偵測、影像分類和實例分割。YOLOv5 的創建者 Ultralytics 也開發了 YOLOv8，與前身相比，YOLOv8 在架構和開發人員體驗方面融入了許多改進和變化。YOLOv8是業界備受推崇的最新最先進模型。

下表比較了五種不同尺寸（以像素為單位）的 YOLOv8 模型的效能指標：YOLOv8n、YOLOv8s、YOLOv8m、YOLOv8l 和 YOLOv8x。這些指標包括驗證資料的不同交集 (IoU) 閾值下的平均精確度 (mAP) 值、採用 ONNX 格式和 A100 TensorRT 的 CPU 推理速度、參數數量以及浮點運算 (FLOP) 數量（分別以百萬和十億為單位）。隨著模型規模的增加，mAP、參數和 FLOP 通常會增加，而速度會降低。YOLOv8x 具有最高的 mAP、參數和 FLOP，但推理速度最慢，而 YOLOv8n 具有最小的尺寸、最快的推理速度和最低的 mAP、參數和 FLOP。
