Tomato Disease Prediction

A Machine Learning project to detect and classify diseases in tomato plants using image data. This project helps farmers and agriculturists identify plant diseases early, improving crop yield and reducing losses.

🔍 Project Overview

Objective: Predict the disease type of tomato leaves from images using machine learning models.

Input: Images of tomato leaves.

Output: Predicted disease class (e.g., Healthy, Early Blight, Late Blight, Leaf Mold, etc.).

Benefits:

Early detection of tomato diseases

Helps farmers take preventive measures

Reduces crop loss and improves yield

📁 Dataset

The dataset consists of labeled images of tomato leaves.

Categories include various tomato diseases and healthy leaves.

Dataset source: [Specify if it's custom or from a public repository like PlantVillage]

🛠️ Technologies Used

Python 3.x

Libraries:

TensorFlow / Keras

OpenCV

NumPy

Pandas

Matplotlib / Seaborn

Machine Learning Concepts:

Convolutional Neural Networks (CNN)

Image preprocessing and augmentation

Model evaluation metrics (Accuracy, Confusion Matrix)






Model Training

Load and preprocess the dataset.

Perform data augmentation to increase dataset variability.

Split dataset into training and testing sets.

Build a CNN model using Keras.

Train the model and monitor performance using validation accuracy.

Evaluate model performance on test data.

📊 Results

Achieved [insert accuracy]% accuracy on test data.

Confusion matrix visualized for model evaluation.

Future Work

Integrate a real-time disease detection system using a mobile app or camera.

Expand the dataset to include more tomato varieties and diseases.

Experiment with transfer learning using pre-trained models like ResNet or EfficientNet.

📄 References

PlantVillage Dataset

Keras and TensorFlow official documentation
Model able to correctly classify multiple tomato diseases and healthy leaves.

