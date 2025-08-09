# ✨ Sign Language Translator – AI-Powered Real-Time Gesture Recognition  

> **Breaking barriers with AI** – Translate sign language into text in real time using computer vision and deep learning.

---

## 🚀 Overview
The **Sign Language Translator** is a deep learning project that uses **MediaPipe** for landmark detection and a **Transformer-based neural network** for gesture classification.  
It allows you to:
- 📸 **Collect** custom sign language gesture data  
- 🧠 **Train** an advanced transformer model  
- 🎥 **Translate** signs into text **live** via your webcam

---

## 🛠 Features
- **Real-time Detection** – Instant sign-to-text translation  
- **Transformer Architecture** – Modern attention-based sequence modeling  
- **Custom Dataset Collection** – Capture your own gestures using `collect_data.py`  
- **Beautiful Visualizations** – Real-time probability bars & styled hand/pose/face landmarks  
- **Three Sample Signs Included** – `hello`, `thanks`, `iloveyou`  

---

## 📂 Project Structure
```
📦 Sign Language Translator
 ┣ 📜 Sign Language Translator.ipynb   # Notebook with model training pipeline
 ┣ 📜 collct_data.py                   # Script for dataset collection
 ┣ 📜 test_1000.py                      # Real-time inference & translation
 ┣ 📜 action_1000.h5                    # Pre-trained weights (optional)
 ┗ 📂 MP_Data                           # Captured gesture data
```

---

## 📸 Data Collection (`collct_data.py`)
Run the following to start collecting gestures:
```bash
python collct_data.py
```
- **Default Actions**: `hello`, `thanks`, `iloveyou`  
- **Sequence Length**: 30 frames per gesture  
- **Camera Index**: Change `cam_index` in the script if your webcam is not `0`.

---

## 🧠 Model Architecture (from `Sign Language Translator.ipynb`)
- **Input**: Sequence of 1662 keypoints (pose, face, hands) over 30 frames  
- **Feature Projection**: Dense layer to embed keypoints  
- **Positional Embedding**: Learned embeddings for sequence positions  
- **Transformer Encoder Block**: Multi-Head Attention + Feed Forward + Residual + LayerNorm  
- **Classification Head**: Dense + Softmax to output gesture probabilities  

---

## 🎯 Real-Time Translation (`test_1000.py`)
Run:
```bash
python test_1000.py
```
- Press **`q`** to quit the feed  
- Adjust `CAMERA_INDEX` if your webcam feed is blank  
- Requires the pre-trained `action_1000.h5` weights  

---

## 📦 Installation
```bash
pip install opencv-python mediapipe tensorflow numpy
```

---

## 🌟 Example Output
When you sign **"hello"**, the system displays:

```
--------------------------------
Predicted: hello   ✅ (0.92)
thanks: 0.05
iloveyou: 0.03
--------------------------------
```
…and overlays styled landmarks with colored probability bars in real time!

---

## 💡 Future Improvements
- Add more sign language gestures  
- Integrate audio output for accessibility  
- Support multiple sign languages

---

## ❤️ Contributing
Pull requests are welcome! Whether it’s adding new signs, improving the model, or enhancing the UI – your help is appreciated.

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
