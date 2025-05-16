# Waste_seggregation_system
A deep learning-based system that classifies waste into Organic and Recyclable categories using CNN and OpenCV. Built with a simple UI using Streamlit.


# 🗂️ Waste Segregation Using Image Classification (CNN)

## 📌 Overview

This project presents a smart waste segregation system that uses a Convolutional Neural Network (CNN) to automatically classify waste images into two main categories:

* **Organic Waste** (e.g., food scraps, biodegradable items)
* **Recyclable Waste** (e.g., plastic bottles, cans)

The project was developed during a one-month internship at Netlip IT Training & Solutions. The aim is to enhance environmental sustainability through intelligent automation of waste sorting, improving the efficiency of disposal and recycling systems.


## 🚀 Technologies Used

* Python 3.x
* TensorFlow & Keras
* OpenCV
* Streamlit
* NumPy & Pandas
* Matplotlib for visualization


## 🧠 CNN Model Summary

* **Model Type:** Sequential CNN
* **Input Image Size:** 150x150 pixels
* **Model Architecture:**

  * Convolutional layers with ReLU activation
  * BatchNormalization for stable learning
  * MaxPooling for dimensionality reduction
  * Dropout for regularization
  * Fully Connected Dense layers
* **Output Layer:** Softmax (2 neurons - Organic or Recyclable)
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam
* **Validation Accuracy:** \~95%


## 📁 Dataset Information

* Structured into 3 directories: `train`, `test`, and `val`
* Each directory has two folders: `Organic/` and `Recyclable/`
* Images are resized and normalized
* Augmentation applied: rotation, flip, zoom

```pytho
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)
```

## 📊 Model Evaluation

```python
model.evaluate(test_generator)
```

* **Accuracy:** \~95% on unseen test data
* **Metrics Used:** Accuracy, Confusion Matrix, Precision, Recall
* **Loss:** Low categorical crossentropy on validation


## 💻 Streamlit Web App Features

* Upload or capture image using webcam
* Real-time prediction of waste type
* Clean and intuitive user interface
* Displays class (Organic/Recyclable) with prediction confidence

```python
st.title("Waste Segregation Classifier")
file = st.file_uploader("Upload Waste Image")
...
prediction = model.predict(img_array)
label = "Organic" if np.argmax(prediction) == 0 else "Recyclable"
st.success(f"Predicted as: {label}")
```


## 📈 Sample Prediction

| Sample         | Predicted Class | Confidence |
| -------------- | --------------- | ---------- |
| Banana Peel    | Organic         | 97%        |
| Plastic Bottle | Recyclable      | 94%        |


## 📎 Folder Structure

```
├── model/
│   └── waste_model.h5
├── streamlit_app/
│   ├── app.py
│   └── utils.py
├── dataset/
│   ├── train/
│   ├── test/
│   └── val/
└── README.md
```


## 💡 Future Enhancements

* Add more waste categories (e.g., E-waste, Hazardous)
* Integrate object detection (YOLOv5)
* Use MobileNet or ResNet for transfer learning
* Deploy on Docker, Heroku, or AWS
* Add Grad-CAM visualization for AI explainability
