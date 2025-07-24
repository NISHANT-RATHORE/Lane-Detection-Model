# CBGA-LaneNet: A Lightweight Model for Real-Time Lane Detection

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/code/nishantrathore123/cbga-lane-detection-final)

A lightweight and efficient lane detection model designed for real-time performance on autonomous driving and ADAS platforms. This model leverages **Conv-BatchNorm-GELU-Attention (CBGA)** modules and an **auxiliary segmentation branch** to achieve a superior balance between accuracy, speed, and model size.

Trained on the **TuSimple** lane detection dataset, CBGA-LaneNet demonstrates high accuracy and robustness under challenging conditions such as shadows, occlusions, and poor lighting.

---

### Demo
<!-- 
  RECOMMENDATION: Create a short GIF showing your model's output on a test video. 
  It's the best way to showcase your project's capabilities.
-->
 
*Lane detection in real-time under various lighting and road conditions.*

---

## 📋 Table of Contents
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
  - [Inference on Images](#inference-on-images)
  - [Inference on Videos](#inference-on-videos)
- [Training](#-training)
- [Dependencies](#-dependencies)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Key Features

*   **Real-time Performance**: Optimized for low-latency inference, making it suitable for deployment on edge devices and embedded automotive hardware.
*   **Lightweight & Efficient**: A small parameter footprint achieved through the use of Ghost convolutions in the segmentation head, minimizing resource requirements.
*   **High Robustness**: Excels in challenging scenarios including:
    *   Heavy shadows and changing lighting.
    *   Partial lane occlusions.
    *   Curved roads and complex lane markings.
    *   Poor visibility conditions.
*   **Advanced Architecture**:
    *   **CBGA Blocks**: Custom blocks combining Convolution, BatchNorm, GELU activation, and Squeeze-and-Excitation (SE) attention for enhanced feature discriminability.
    *   **Attention Mechanism**: SE attention allows the model to adaptively prioritize the most relevant features for lane detection.
*   **Auxiliary Training for Improved Accuracy**: An auxiliary segmentation branch is used during training to improve gradient flow and lane continuity, without adding any overhead to the final inference model.

---

## 🏗️ Model Architecture

The model's architecture is designed for efficiency and accuracy. It consists of the following key components:

1.  **Backbone (Feature Extractor)**: A ResNet-based backbone is used for robust initial feature extraction from the input image.

2.  **CBGA Blocks**: The core of the model. These blocks process the features from the backbone.
    ```
    Input -> Conv2D -> BatchNorm -> GELU Activation -> Squeeze-and-Excitation -> Output
    ```
    The SE Attention module helps the network focus on subtle yet crucial lane features while suppressing irrelevant background noise.

3.  **Main Segmentation Head**: This head predicts the final lane markings. It utilizes lightweight **Ghost convolutions** to reduce the number of parameters and computational cost (FLOPs) without sacrificing performance.

4.  **Auxiliary Segmentation Branch (Training Only)**: This branch is attached to an intermediate layer of the network during training. It provides an additional supervision signal, which helps combat vanishing gradients and encourages the model to learn more representative features early on. It is detached before deployment, ensuring no impact on inference speed.

```
                  +--------------------------------+
[Input Image] ->  |      ResNet Backbone           |
                  +--------------------------------+
                            |
                            v
                  +--------------------------------+
                  |  CBGA Blocks (with SE Attn)    |
                  +--------------------------------+
                            |                       \
                            v                        \ (Training Only)
              +----------------------------+          +------------------------+
              |   Main Head (GhostConv)    |          |  Auxiliary Head        |
              +----------------------------+          +------------------------+
                            |                                  |
                            v                                  v
                    [Final Lane Output]               [Auxiliary Loss]
```

---

## 🚀 Performance

The model was evaluated on the **TuSimple lane detection dataset**. It achieves competitive accuracy while significantly outperforming heavier models in inference speed and parameter count.

| Metric              | Value                  | Notes                                                    |
| ------------------- | ---------------------- | -------------------------------------------------------- |
| **Accuracy**        | **XX.XX%**             | Fill in with your official TuSimple accuracy metric.     |
| **Inference Speed** | **~XX FPS**            | Fill in with your FPS (e.g., on a V100, Jetson, or CPU). |
| **Model Parameters**| **X.X M**              | Fill in with the total number of model parameters.       |
| **Model Size**      | **X.X MB**             | Fill in with the size of the final `.pth` or `.onnx` file. |

This balance makes the model an ideal candidate for systems where both precision and processing speed are critical.

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cbga-lanenet.git
    cd cbga-lanenet
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Usage

### Download Pre-trained Weights

Download the trained model weights and place them in the `weights/` directory.

[**Download Pre-trained Weights Here**](https://example.com/link-to-your-weights.pth) <!--- Host your .pth file and link it here --->

### Inference on Images

To run lane detection on a single image:
```bash
python detect.py --source /path/to/your/image.jpg --weights weights/cbga_lanenet.pth
```
The output image with the detected lanes will be saved in the `runs/detect/` directory.

### Inference on Videos

To run lane detection on a video file:
```bash
python detect.py --source /path/to/your/video.mp4 --weights weights/cbga_lanenet.pth
```
The processed video will be saved in the `runs/detect/` directory.

---

## 🧠 Training

To train the model on the TuSimple dataset from scratch:

1.  **Download the Dataset**: Download the TuSimple lane detection dataset and organize it according to the expected structure. You may need to create a `.yaml` file to specify dataset paths.

    ```yaml
    # Example: tusimple.yaml
    train: /path/to/tusimple/train_set/
    val: /path/to/tusimple/test_set/
    
    # number of classes
    nc: 1 # Or however many classes you are predicting
    
    # class names
    names: [ 'lane' ]
    ```

2.  **Start Training**: Run the training script. Specify the model configuration file, dataset configuration file, and desired batch size.

    ```bash
    python train.py --cfg models/cbga_model.yaml --data tusimple.yaml --batch-size 16 --epochs 100
    ```

---

## 📦 Dependencies

*   Python 3.8+
*   PyTorch (>=1.8)
*   OpenCV-Python
*   NumPy
*   tqdm
*   (Add any other specific libraries here)

All dependencies are listed in `requirements.txt`.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✍️ Citation

If you use this model or code in your research, please consider citing it:

```bibtex
@misc{yourname2023cbgalanenet,
    author    = {Your Name},
    title     = {CBGA-LaneNet: A Lightweight Model for Real-Time Lane Detection},
    year      = {2023},
    publisher = {GitHub},
    journal   = {GitHub repository},
    howpublished = {\url{https://github.com/your-username/cbga-lanenet}}
}
```

---

## 🙏 Acknowledgements

*   This work was trained and validated using the [TuSimple Lane Detection Dataset](https://github.com/TuSimple/tusimple-benchmark).
*   Inspiration from the ResNet, Squeeze-and-Excitation, and GhostNet architectures.
*   Thanks to the Kaggle community for providing a great platform for experimentation.
