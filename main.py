import os
import sys
from A.Random_forest import run_random_forest
from A.SVM import run_svm
from B.CNN import run_cnn

def main():
    print("Starting Medical Image Classification Project...")
    # Ensure dataset folder and required files exist
    required_files = [
        "Datasets/BreastMNIST.npz",
        "Datasets/bloodmnist.npz"
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file `{file}` not found.")
            print("Please ensure the Datasets folder contains the following files:")
            print("  - BreastMNIST.npz")
            print("  - bloodmnist.npz")
            sys.exit(1)
    # Task A - Binary Classification
    print("\nRunning Task A: Binary Classification (BreastMNIST)...")
    print("\n--- Running Random Forest ---")
    run_random_forest()
    print("\n--- Running Support Vector Machine (SVM) ---")
    run_svm()
    
    # Task B - Multi-class Classification
    print("\nRunning Task B: Multi-class Classification (BloodMNIST)...")
    run_cnn()
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()