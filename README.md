# AMLS_24_25_SN21008931

This project applies machine learning techniques to classify medical images using the BreastMNIST and BloodMNIST datasets from MedMNIST. The tasks are as follows:

- **Task A**: Binary classification of breast cancer (BreastMNIST) using Random Forest and Support Vector Machine (SVM).  
- **Task B**: Multi-class classification of blood cells (BloodMNIST) using a Convolutional Neural Network (CNN).  

The implementation includes preprocessing, data augmentation, hyperparameter tuning, and result visualization. The project is structured for ease of reproducibility and understanding.

---

## File Structure

### **A/**
- **`Random_forest.py`**: Python script implementing Random Forest for Task A.
- **`SVM.py`**: Python script implementing Support Vector Machine (SVM) for Task A.
- **`random_forest_accuracy_vs_n_estimators.png`**: Plot showing Random Forest accuracy vs. the number of trees.

### **B/**
- **`CNN.py`**: Python script implementing CNN for Task B.
- **`confusion_matrix_14epochs.png`**: Confusion matrix of CNN predictions after 14 epochs.
- **`training_validation_curves_30epochs.png`**: Training and validation accuracy/loss curves for CNN after 30 epochs.

### **Datasets/**
- Empty folder. Place the following datasets here:
  - `breastmnist.npz` (for Task A)
  - `bloodmnist.npz` (for Task B)  

Both datasets can be downloaded from [MedMNIST](https://medmnist.com/).

### **`main.py`**
- Script to run both tasks:
  - Executes `Random_forest.py` and `SVM.py` for Task A.
  - Executes `CNN.py` for Task B using pre-configured parameters.

### **`README.md`**
- Instructions and details about the project (this file).

---

## How to Run

### 1. Download Datasets
- Obtain the datasets `breastmnist.npz` and `bloodmnist.npz` from [MedMNIST](https://medmnist.com/).
- Place the datasets in the `Datasets/` folder.

### 2. Set Up Environment
Install Python 3.11.6 and the required libraries. Use the following commands to set up the environment:
```bash
pip install numpy==1.26.2 scikit-learn==1.3.2 tensorflow==2.15.0 seaborn==0.13.0 matplotlib==3.8.
2 medmnist==2.0.0 keras-tuner==1.4.4
```

### 3. Run the Project

Run the `main.py` script to execute both tasks:
```bash
python main.py
```

By default:
- **Task A**: Evaluates Random Forest and SVM with hyperparameter tuning and visualization of results.
- **Task B**: Evaluates the CNN model with hyperparameter tuning for 9 epochs initially, followed by 30 epochs to see the learning curves and finally choose 14 for final training.

---

## 4. View Results
- During execution, visualizations such as accuracy curves and confusion matrices will be displayed as well as the text output for validation accuracy and test accuracy. Close these visualizations to allow the script to continue.
- All results will be saved in their respective folders (`A/` or `B/`).