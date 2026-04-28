![](TrafficTracker-logo.png)
# 🚦 Real-Time Dynamic Lane Congestion Interpretation System (YOLOv8 + DLCI)

A real-time traffic analysis system that performs vehicle detection, tracking, and lane-wise congestion estimation using computer vision and deep learning techniques.

---

## 📌 Overview

This project analyzes traffic videos in real time using **YOLOv8** for vehicle detection and tracking. It introduces a novel metric called **Dynamic Lane Congestion Index (DLCI)**, which combines vehicle density and speed to provide accurate congestion estimation at the lane level.

Unlike traditional systems that rely only on vehicle counting, this system performs **lane-wise multi-parameter analysis**, making it more realistic and effective.

---

## 🔥 Features

- Real-time vehicle detection using YOLOv8  
- Built-in tracking (no DeepSORT required)  
- Lane-wise traffic analysis  
- Vehicle density estimation (weighted by class)  
- Speed estimation using centroid displacement  
- DLCI-based congestion calculation  
- Congestion classification (Low / Medium / High)  
- Visualization using color-coded lanes  

---

## 🧠 Methodology

1. Input traffic video (dataset or webcam)  
2. Preprocessing using OpenCV  
3. Vehicle detection & tracking using YOLOv8  
4. Lane segmentation using geometric division  
5. Assign vehicles to lanes (centroid-based)  
6. Calculate vehicle density (weighted)  
7. Estimate speed using centroid displacement  
8. Compute DLCI  
9. Classify congestion levels  

---

## ⚙️ Technologies Used

- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- PyTorch  
- NumPy  
- Matplotlib  
- Pandas  

---

## 🚀 Installation

```bash
pip install -r requirements.txt