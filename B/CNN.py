import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras_tuner as kt
import os

def run_cnn():
        
    # --------------------------------------------------
    # 1. Load and preprocess data
    # --------------------------------------------------
    data_path = 'Datasets/bloodmnist.npz'  
    bloodmnist = np.load(data_path)

    train_images = bloodmnist['train_images']
    train_labels = bloodmnist['train_labels']
    val_images = bloodmnist['val_images']
    val_labels = bloodmnist['val_labels']
    test_images = bloodmnist['test_images']
    test_labels = bloodmnist['test_labels']

    # Normalize the images (scale from [0,255] to [0,1])
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # One-hot encode the labels
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
    test_labels = to_categorical(test_labels)

    # Set a random seed for reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Number of classes
    num_classes = train_labels.shape[1]

    # --------------------------------------------------
    # 2. Define the hyperparameter search model
    # --------------------------------------------------
    def build_model(hp):
        """
        Build a CNN model with hyperparameters searched by Keras Tuner.
        
        Parameters
        ----------
        hp : HyperParameters
            A HyperParameters instance to manage hyperparameter tuning.
        
        Returns
        -------
        model : Sequential
            A compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Input(shape=(28, 28, 3)))

        # Tuning the number of filters and kernel size in the first convolution
        hp_filters_1 = hp.Int('filters_1', min_value=80, max_value=160, step=16)
        hp_kernel_size_1 = hp.Choice('kernel_size_1', values=[3, 5])

        # Tuning the number of filters and kernel size in the second convolution
        hp_filters_2 = hp.Int('filters_2', min_value=160, max_value=384, step=32)
        hp_kernel_size_2 = hp.Choice('kernel_size_2', values=[3, 5])
        
        # First Convolution + Pooling
        model.add(Conv2D(filters=hp_filters_1,
                        kernel_size=(hp_kernel_size_1, hp_kernel_size_1),
                        activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # Second Convolution + Pooling
        model.add(Conv2D(filters=hp_filters_2,
                        kernel_size=(hp_kernel_size_2, hp_kernel_size_2),
                        activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # Flatten the output
        model.add(Flatten())

        # Tuning the number of units in the Dense layer
        hp_units = hp.Int('units', min_value=256, max_value=448, step=64)
        model.add(Dense(units=hp_units, activation='relu'))

        # Tuning the Dropout rate (0.1 or 0.3)
        hp_dropout = hp.Choice('dropout', values=[0.1, 0.3])
        model.add(Dropout(rate=hp_dropout))

        # Output layer with 'softmax' for multi-class classification
        model.add(Dense(num_classes, activation='softmax'))

        # Tuning the learning rate (1e-3 or 1e-4)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    # --------------------------------------------------
    # 3. Hyperparameter search with Keras Tuner
    # --------------------------------------------------
    # RandomSearch tuner to explore the hyperparameter space
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=20,  # Number of different hyperparameter configurations to try
        executions_per_trial=1,
        directory='hyperparam_tuning_3',
        project_name='bloodmnist_tuning_dropout'
    )

    # Run the hyperparameter search
    tuner.search(train_images, train_labels, 
                epochs=9,  # Early-phase short training
                validation_data=(val_images, val_labels),
                verbose=1)

    # --------------------------------------------------
    # 4. Retrieve the best hyperparameters
    # --------------------------------------------------
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best Hyperparameters:")
    print("filters_1:", best_hps.get('filters_1'))
    print("kernel_size_1:", best_hps.get('kernel_size_1'))
    print("filters_2:", best_hps.get('filters_2'))
    print("kernel_size_2:", best_hps.get('kernel_size_2'))
    print("units:", best_hps.get('units'))
    print("dropout:", best_hps.get('dropout'))
    print("learning_rate:", best_hps.get('learning_rate'))
    # --------------------------------------------------
    # 5. Build and train the initial model for 30 epochs
    # --------------------------------------------------
    model = tuner.hypermodel.build(best_hps)
    history_30 = model.fit(
        train_images, train_labels, 
        epochs=30,
        validation_data=(val_images, val_labels),
        verbose=1
    )

    # --- Training & Validation Accuracy and Loss Curves ---
    acc = history_30.history['accuracy']
    val_acc = history_30.history['val_accuracy']
    loss = history_30.history['loss']
    val_loss = history_30.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy (30 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss (30 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    # Save the training curves figure
    training_curves_path = "training_validation_curves_30epochs.png"
    plt.savefig(training_curves_path)
    print(f"Training curves figure saved to {training_curves_path}.")

    # Display the training curves
    plt.show(block=False)

    # --------------------------------------------------
    # 6. Retrain the model for 14 epochs based on observations
    # --------------------------------------------------
    model = tuner.hypermodel.build(best_hps)
    history_14 = model.fit(
        train_images, train_labels, 
        epochs=14,
        validation_data=(val_images, val_labels),
        verbose=1
    )

    # --------------------------------------------------
    # 7. Evaluate the model on the test set
    # --------------------------------------------------
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print("Test Accuracy: {:.2f}%".format(test_acc * 100))

    # Generate predictions for the test set
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Save the confusion matrix figure
    cm_fig_path = "confusion_matrix_14epochs.png"
    plt.savefig(cm_fig_path)
    print(f"Confusion matrix saved to {cm_fig_path}.")

    # Display the confusion matrix
    plt.show(block=False)
    print("CNN completed. Results saved to `B/` folder.")