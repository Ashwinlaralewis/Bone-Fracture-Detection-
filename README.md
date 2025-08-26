🦴 Bone Fracture Detection using CNN

This project uses Convolutional Neural Networks (CNNs) to detect bone fractures from X-ray images. The system includes training scripts, prediction utilities, and a simple GUI for testing.

📌 Features

CNN-based deep learning model for fracture detection

Training and testing on X-ray datasets

Pre-trained weights included

GUI for image-based predictions

Accuracy visualization with plots

📂 Project Structure
Bone-Fracture-Detection-/
│── Dataset/               # X-ray dataset
│── weights/               # Pre-trained model weights
│── plots/                 # Training/validation accuracy and loss plots
│── images/                # Sample images
│── test/                  # Test dataset
│── mainGUI.py             # GUI for predictions
│── training_fracture.py   # Training model for fracture detection
│── training_parts.py      # Training model for bone parts classification
│── predictions.py         # Run predictions on images
│── prediction_test.py     # Script to test predictions
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

⚙️ Installation
1. Clone the Repository
git clone https://github.com/Ashwinlaralewis/Bone-Fracture-Detection-.git
cd Bone-Fracture-Detection-

2. Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3. Install Dependencies
pip install -r requirements.txt

🚀 Usage
1. Train the Model

For fracture detection:

python training_fracture.py


For bone part classification:

python training_parts.py

2. Test Predictions
python prediction_test.py

3. Run GUI for Image Prediction
python mainGUI.py

📊 Results

Model achieves high accuracy in detecting fractures from bone X-ray images.

Training/validation accuracy and loss curves are available in the plots/ folder.

📦 Requirements

Key dependencies (see requirements.txt for full list):

Python 3.8+

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

📸 Sample Predictions
X-ray Input	Prediction

	Fracture Detected

	Normal
📌 Future Improvements

Improve dataset quality & size

Use transfer learning (ResNet, DenseNet, EfficientNet)

Deploy as a web application

Integrate Grad-CAM for explainable AI

👨‍💻 Author

Developed by Ashwin Lara Lewis
