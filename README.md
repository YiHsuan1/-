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


