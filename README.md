## Tomato Disease Prediction using Deep Learning

A **Deep Learning project** that detects and classifies diseases in tomato plant leaves using image data.  
This system enables **early disease detection**, helping farmers and agronomists take timely action to **reduce crop loss** and **improve yield**.

---

## Project Overview

**Objective:**  
To automatically predict the disease type of tomato leaves from images using a Convolutional Neural Network (CNN).

**Input:**  
Images of tomato leaves.

**Output:**  
Predicted disease class such as:  
- **Healthy**  
- **Early Blight**  
- **Late Blight**  
- **Leaf Mold**  
- *(and more, depending on dataset categories)*

**Key Benefits:**
- Enables **early detection** of tomato diseases.  
- Assists farmers in **taking preventive actions**.  
- Minimizes **economic losses** and boosts productivity.

---

## Dataset

- The dataset consists of labeled images of tomato leaves.  
- Each class corresponds to a specific tomato disease or healthy leaf.  
- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) *(or mention if custom)*  
- **Structure:**
  ```
  Tomato_Cases/
  ├── Healthy/
  ├── Early_Blight/
  ├── Late_Blight/
  └── Leaf_Mold/
  ```

---

## Methodology

1. **Data Loading:**  
   Used `tf.keras.preprocessing.image_dataset_from_directory()` to load images directly from folders.

2. **Preprocessing & Augmentation:**  
   Applied layers for improved generalization:  
   - Random Flip  
   - Random Rotation  
   - Random Zoom  
   - Random Contrast  
   - Random Translation  
   - Resizing & Rescaling  

3. **Model Architecture (Custom CNN):**
   ```
   Rescaling → Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense → Dense (Softmax)
   ```
   - Built using **Keras Sequential API**.  
   - Tuned for 50 epochs with **Adam optimizer** and **categorical cross-entropy loss**.

4. **Training:**  
   - Batch Size: 32  
   - Image Size: 256×256×3  
   - Epochs: 50  
   - Validation Split: [Insert %]  
   - Monitored accuracy and loss curves during training.

5. **Evaluation:**  
   - Tested on unseen data using `model.evaluate()`  
   - Visualized **Confusion Matrix**, **Accuracy**, and **Loss Curves**.

---

## Results

| Metric | Result |
|--------|---------|
| **Training Accuracy** | [97.22]% |
| **Validation Accuracy** | [89.63]% |
| **Test Accuracy** | [90.81]% |
| **Loss** | [0.0821] |

**Visualization:** 
- Accuracy vs. Epochs  
- Loss vs. Epochs
- <img width="826" height="834" alt="image" src="https://github.com/user-attachments/assets/47d1f4dc-b0be-4047-b5f9-cc56e6a9f8c4" />

---

## Technologies Used

**Libraries & Frameworks:**
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  

**Machine Learning Concepts:**
- Convolutional Neural Networks (CNN)  
- Image Preprocessing & Augmentation  
- Model Evaluation (Accuracy, Confusion Matrix)

## Future Work

- Integrate with a **real-time detection system** using a webcam or mobile app.  
- Experiment with **Transfer Learning** using pre-trained models like ResNet, EfficientNet, or MobileNet.  
- Deploy the model as a **web dashboard or mobile API**.  
- Expand dataset to include more tomato species and leaf conditions.


---

