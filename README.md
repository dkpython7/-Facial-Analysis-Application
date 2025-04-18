# 🤖 Facial Analysis Application

Real-time facial detection system with **emotion recognition**, **age prediction**, and **head pose estimation** using Python, OpenCV, and machine learning models.

---

## 📁 Project Structure

```
facial_analysis/
├── models/               
│   ├── emotion_model.pkl      # Emotion detection model
│   ├── scaler.pkl             # Scaler for emotion model
│   └── age_model.h5           # Deep learning age prediction model
├── facial_analysis.py         # Main application script
└── README.md                  # Project documentation
```

---

## 🚀 Features

- 🎭 Emotion Detection (Happy, Sad, Angry, Surprise, Neutral, etc.)
- 👶 Age Prediction (Predicts age from face using CNN)
- 🧠 Head Pose Estimation (Detects head orientation: left, right, up, down)
- 📹 Real-time webcam video processing
- 🛠 Auto fallback for models if unavailable

---

## 🧰 Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- scikit-learn
- NumPy

---

## 🛠 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/dkpython7/-Facial-Analysis-Application.git
cd -Facial-Analysis-Application/facial_analysis
```

### 2. Install Required Libraries

```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy
```

### 3. Run the Application

```bash
python facial_analysis.py
```

Press `q` to exit the webcam window.

---

## 📌 Notes

- All models are pre-trained and saved in the `models/` folder.
- If models are not found, fallback options will auto-generate basic models.
- Works with most standard webcams.

---

## 📈 Future Enhancements

- Add gender recognition
- Multi-face tracking support
- Integration with GUI dashboard
- Save predictions in a local database

---

## 👨‍💻 Author

**Md Imran**  
[GitHub](https://github.com/dkpython7) • [LinkedIn](https://www.linkedin.com/in/md-imran-48a443292)

---
