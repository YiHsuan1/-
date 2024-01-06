# Efficient Object Detection with YOLOV8 and KerasCV（使用YOLOV8和KerasCV進行高效能物件偵測）
南華大學跨領域-人工智慧期中報告 11123024楊佳宜 11123007陳奕瑄 11118128 吳佳恩

介紹
KerasCV 是 Keras 用於電腦視覺任務的擴展。在此範例中，我們將了解如何使用 KerasCV 訓練 YOLOV8 目標偵測模型。
KerasCV 包含針對流行電腦視覺資料集的預訓練模型，例如 ImageNet、COCO 和 Pascal VOC，可用於遷移學習。KerasCV 還提供了一系列視覺化工具，用於檢查模型學習的中間表示以及可視化物件偵測和分割任務的結果。
如果您有興趣了解使用 KerasCV 進行物件偵測，我強烈建議您查看 lukewood 建立的指南。此資源可在 使用 KerasCV 進行物件偵測中取得，全面概述了使用 KerasCV 建構物件偵測模型所需的基本概念和技術。
```python
!pip install —upgrade git+https://github.com/keras-team/keras-cv -q
```
