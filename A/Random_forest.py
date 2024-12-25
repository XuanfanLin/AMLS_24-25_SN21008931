import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.ndimage import rotate
import cv2
from sklearn.model_selection import cross_val_score


def run_random_forest():
    # --------------------------------------------------
    # 1. Load data and preprocess
    # --------------------------------------------------

    # Path to the BreastMNIST dataset file
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

    # Normalize the images (pixel values from 0 to 1)
    train_images = train_images / 255.0
    val_images = val_images / 255.0
    test_images = test_images / 255.0

    # --------------------------------------------------
    # 2. Define the data augmentation function
    # --------------------------------------------------
    def augment_images(images):
        """
        Perform data augmentation on the input image array, including:
        1. Keeping the original image
        2. Rotating the image by ±15°
        3. Flipping the image horizontally
        4. Applying Gaussian blur

        Parameters
        ----------
        images : numpy.ndarray
            Original array of images, typically of shape (num_samples, height, width).

        Returns
        -------
        numpy.ndarray
            The augmented images array. The number of images after augmentation is 
            several times larger than the original.
        """
        augmented_images = []
        for img in images:
            # Original image
            augmented_images.append(img)
            
            # Rotate ±15° (reshape=False keeps the same shape)
            augmented_images.append(rotate(img, angle=15, reshape=False, mode='nearest'))
            augmented_images.append(rotate(img, angle=-15, reshape=False, mode='nearest'))
            
            # Horizontal flip
            augmented_images.append(np.fliplr(img))
            
            # Gaussian blur
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            augmented_images.append(blurred)
        
        return np.array(augmented_images)

    # --------------------------------------------------
    # 2.1 Apply data augmentation on the training set
    # --------------------------------------------------
    augmented_train_images = augment_images(train_images)

    # Calculate how many augmented images are created per original image
    num_augmentations = len(augmented_train_images) // len(train_images)

    # Repeat the training labels to match the augmented images
    augmented_train_labels = np.repeat(train_labels, num_augmentations)

    # Shuffle the augmented dataset
    augmented_train_images, augmented_train_labels = shuffle(
        augmented_train_images, augmented_train_labels, random_state=42
    )

    # --------------------------------------------------
    # 3. Prepare training, validation, and test sets
    # --------------------------------------------------
    # Flatten images (since RandomForest typically takes 2D features)
    X_train = augmented_train_images.reshape(len(augmented_train_images), -1)
    X_val = val_images.reshape(len(val_images), -1)
    X_test = test_images.reshape(len(test_images), -1)

    # Convert labels to 1D arrays
    y_train = augmented_train_labels.ravel()
    val_labels = val_labels.ravel()
    test_labels = test_labels.ravel()

    # --------------------------------------------------
    # 4. Model training and selection
    # --------------------------------------------------
    # Define the range of n_estimators (25, 50, 75, ..., 400)
    n_estimators_range = range(25, 401, 25)

    # Lists to store validation and cross-validation accuracies
    val_accuracies = []
    cv_accuracies = []

    # Loop over possible values of n_estimators
    for n_estimators in n_estimators_range:
        # Define a random forest classifier
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        # Train the classifier on the augmented dataset
        clf.fit(X_train, y_train)
        
        # Predict on the validation set and compute accuracy
        predicted_val_labels = clf.predict(X_val)
        val_accuracy = accuracy_score(val_labels, predicted_val_labels)
        val_accuracies.append(val_accuracy)
        
        # Perform 5-fold cross-validation on the training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        cv_accuracies.append(cv_scores.mean())
        
        # Print out the results
        print(f"n_estimators: {n_estimators}, "
            f"Validation Accuracy: {val_accuracy:.4f}, "
            f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

    # Compute the combined score (simple sum of validation and cross-validation accuracies)
    combined_scores = np.array(val_accuracies) + np.array(cv_accuracies)

    # --------------------------------------------------
    # 5. Visualization
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Plot Validation Accuracy and Cross-validation Accuracy
    plt.plot(n_estimators_range, val_accuracies, marker='o', label='Validation Accuracy')
    plt.plot(n_estimators_range, cv_accuracies, marker='o', label='Cross-validation Accuracy', color='orange')

    # Set labels and title
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Estimators (Random Forest)')
    plt.grid(True)
    plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("random_forest_accuracy_vs_n_estimators.png")

    # Show the plot
    plt.show(block=False)

    # --------------------------------------------------
    # 6. Select the best model and evaluate on the test set
    # --------------------------------------------------
    # Find the best n_estimators based on the combined scores
    best_n_estimators = n_estimators_range[np.argmax(combined_scores)]
    print(f"Best n_estimators based on combined score: {best_n_estimators}")

    # Train the final model using the best n_estimators
    clf_final = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
    clf_final.fit(X_train, y_train)

    # Predict on the test set
    predicted_labels = clf_final.predict(X_test)

    # Evaluate with 5-fold cross-validation on the training set
    scores = cross_val_score(clf_final, X_train, y_train, cv=5)
    print("Average cross-validation score: ", scores.mean())

    # Evaluate on the test set
    test_accuracy = accuracy_score(test_labels, predicted_labels)
    print("The Accuracy on the test dataset:", test_accuracy)

    print("Random Forest completed. Results saved to `A/` folder.")