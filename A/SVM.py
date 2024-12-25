import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import rotate
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def run_svm():
        
    # --------------------------------------------------
    # 1. Load and preprocess data
    # --------------------------------------------------
    # Path to the BreastMNIST dataset
    data_path = 'Datasets/BreastMNIST.npz'

    # Load the dataset
    BreastMNIST = np.load(data_path)

    # Separate the dataset into training, validation, and test sets
    train_images = BreastMNIST['train_images']
    train_labels = BreastMNIST['train_labels']
    val_images = BreastMNIST['val_images']
    val_labels = BreastMNIST['val_labels']
    test_images = BreastMNIST['test_images']
    test_labels = BreastMNIST['test_labels']

    # Normalize the images to the range [0, 1]
    train_images = train_images / 255.0
    val_images = val_images / 255.0
    test_images = test_images / 255.0

    # --------------------------------------------------
    # 2. Define data augmentation function
    # --------------------------------------------------
    def augment_images(images):
        """
        Augment the input images with various transformations:
        1. Keep the original image.
        2. Rotate by +15° and -15°.
        3. Flip the image horizontally.
        4. Apply Gaussian blur.

        Parameters
        ----------
        images : numpy.ndarray
            Original images, typically of shape (num_samples, height, width).

        Returns
        -------
        numpy.ndarray
            Augmented images with more samples than the original input.
        """
        augmented_images = []
        for img in images:
            # Original image
            augmented_images.append(img)
            
            # Rotate images by ±15 degrees
            augmented_images.append(rotate(img, angle=15, reshape=False, mode='nearest'))
            augmented_images.append(rotate(img, angle=-15, reshape=False, mode='nearest'))
            
            # Flip the image horizontally
            augmented_images.append(np.fliplr(img))
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            augmented_images.append(blurred)
        
        return np.array(augmented_images)

    # --------------------------------------------------
    # 2.1 Apply augmentation to training images
    # --------------------------------------------------
    augmented_train_images = augment_images(train_images)

    # Calculate number of augmentations per original image
    num_augmentations = len(augmented_train_images) // len(train_images)

    # Repeat labels to match the number of augmented images
    augmented_train_labels = np.repeat(train_labels, num_augmentations)

    # Shuffle the augmented dataset
    augmented_train_images, augmented_train_labels = shuffle(
        augmented_train_images, augmented_train_labels, random_state=42
    )

    # --------------------------------------------------
    # 3. Prepare training, validation, and test sets
    # --------------------------------------------------
    # Flatten images for SVM input
    X_train = augmented_train_images.reshape(len(augmented_train_images), -1)
    X_val = val_images.reshape(len(val_images), -1)
    X_test = test_images.reshape(len(test_images), -1)

    # Convert labels to 1D arrays
    y_train = augmented_train_labels.ravel()
    val_labels = val_labels.ravel()
    test_labels = test_labels.ravel()

    # --------------------------------------------------
    # 4. Set up GridSearchCV for SVM
    # --------------------------------------------------
    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],     # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]      # Kernel coefficient
    }

    # Create the SVM model
    svm_model = SVC(random_state=42)

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    # --------------------------------------------------
    # 5. Train and find the best model
    # --------------------------------------------------
    # Fit the model (with grid search) on the augmented training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best model
    best_svm_model = grid_search.best_estimator_

    # Print out the best parameters and their CV accuracy
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # --------------------------------------------------
    # 6. Evaluate the best model on the validation and test sets
    # --------------------------------------------------
    val_predictions = best_svm_model.predict(X_val)
    test_predictions = best_svm_model.predict(X_test)

    # Calculate accuracy
    val_accuracy = accuracy_score(val_labels, val_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f"Validation Accuracy with Best Model: {val_accuracy:.4f}")
    print(f"Test Accuracy with Best Model: {test_accuracy:.4f}")
    print("SVM completed. Results saved to `A/` folder.")