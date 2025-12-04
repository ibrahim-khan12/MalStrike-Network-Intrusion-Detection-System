# Network Intrusion Detection System

A comprehensive machine learning-based Network Intrusion Detection System (NIDS) implementing multiple classification approaches for detecting malicious network traffic and cyber attacks.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Overview

MalStrike is an advanced Network Intrusion Detection System that employs multiple machine learning approaches to identify and classify network attacks. The project implements both binary and multi-class classification models, along with a novel Morphological Extreme Learning Machine (ELM) approach for efficient malware detection.

## ✨ Features

- **Multiple Classification Approaches**: Binary and multi-class network intrusion detection
- **Advanced ML Models**: XGBoost, Deep Neural Networks, and Morphological ELM
- **Web Interface**: Interactive web application for real-time inference
- **Comprehensive Evaluation**: Detailed performance metrics including accuracy, precision, recall, F1-score, and confusion matrices
- **Bias-Variance Analysis**: In-depth model generalization assessment
- **Feature Engineering**: IP address octet extraction and feature scaling

## 🤖 Models Implemented

### 1. Morphological Extreme Learning Machine (BaseModel)
- Custom implementation of Morphological ELM
- Uses morphological dilation operations in hidden layer
- Fast training with pseudoinverse computation
- Model export to JSON for browser-based inference

### 2. Binary Classification (MalStrike_Binary2)
- **Algorithm**: XGBoost Classifier
- **Task**: Benign vs Attack detection
- **Features**:
  - Stratified train-test split
  - Class imbalance handling through downsampling
  - Hyperparameter tuning for optimal performance
  - Bias-variance decomposition analysis

### 3. Multi-Class Classification (MalStrike_Multi)
- **Algorithm**: Deep Neural Network (TensorFlow/Keras)
- **Task**: Multi-class attack type detection
- **Architecture**:
  - Input layer with feature scaling
  - 3 hidden layers (256-256-128 neurons)
  - Batch normalization and dropout for regularization
  - Softmax output layer
- **Training Features**:
  - Early stopping
  - Learning rate reduction on plateau
  - Class weight balancing
  - Model checkpointing

## 📊 Dataset

The project uses the **OD-IDS2022 Dataset** for network intrusion detection, which includes:
- Network traffic features (ports, protocols, packet statistics)
- Source and destination IP addresses
- Multiple attack types and benign traffic
- High-dimensional feature space

## 📁 Project Structure

```
MalStrike-IDS/
│
├── BaseModel.ipynb                 # Morphological ELM implementation
├── MalStrike_Binary2.ipynb        # Binary classification (XGBoost)
├── MalStrike_Multi.ipynb          # Multi-class classification (DNN)
├── ML-Article-Project/            # Web interface
│   ├── index.html                 # Frontend interface
│   ├── script.js                  # Model inference logic
│   ├── style.css                  # Styling
│   └── melm_model.json           # Exported model weights
├── i227436_Project1_CY-B.pdf     # Project documentation
├── i227436_Project_CY-B.pdf      # Additional documentation
└── README.md                      # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Modern web browser (for web interface)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MalStrike-IDS.git
cd MalStrike-IDS
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
```

3. For Google Colab users:
```python
!pip install xgboost==1.7.6 seaborn==0.13.2
```

## 💻 Usage

### Running Notebooks

#### BaseModel (Morphological ELM)
```bash
jupyter notebook BaseModel.ipynb
```
- Upload your dataset
- Train the Morphological ELM model
- Export model to JSON for web inference

#### Binary Classification
```bash
jupyter notebook MalStrike_Binary2.ipynb
```
- Loads OD-IDS2022 dataset
- Preprocesses features
- Trains XGBoost classifier
- Evaluates binary classification performance

#### Multi-Class Classification
```bash
jupyter notebook MalStrike_Multi.ipynb
```
- Preprocesses multi-class data
- Trains deep neural network
- Visualizes training curves
- Generates comprehensive evaluation metrics

### Web Interface

1. Navigate to `ML-Article-Project/`
2. Open `index.html` in a web browser
3. Upload model JSON file
4. Input network features for real-time prediction

## 📈 Results

### Binary Classification Performance
- **Algorithm**: XGBoost
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion Matrix, ROC curves

### Multi-Class Classification Performance
- **Algorithm**: Deep Neural Network
- **Metrics**: Macro/Weighted Precision, Recall, F1-Score
- **Analysis**: ROC-AUC (OVR), Feature variance, Correlation analysis
- **Visualization**: Training curves, Confusion Matrix heatmap

### Model Analysis
- Bias-variance decomposition
- Feature importance analysis
- Covariance and correlation studies

## 🛠 Technologies Used

- **Python**: Core programming language
- **NumPy & Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities and metrics
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning framework
- **Matplotlib & Seaborn**: Data visualization
- **HTML/CSS/JavaScript**: Web interface

## 👥 Contributors

- Student IDs: 22I-1755
- Course: Machine Learning (ML-B)
- Semester: 7th

## 📄 License

This project is developed as part of an academic assignment. Please refer to your institution's academic integrity policies before using or distributing this code.

## 🙏 Acknowledgments

- Dataset: OD-IDS2022 Network Intrusion Detection Dataset
- Inspiration: Research papers on Morphological Neural Networks and ELM
- Course instructors and teaching assistants

---

**Note**: For detailed implementation information, please refer to the Jupyter notebooks and project documentation PDFs included in this repository.

## 📞 Contact

For questions or collaborations, please open an issue in this repository.

---

*Last Updated: December 2025*
