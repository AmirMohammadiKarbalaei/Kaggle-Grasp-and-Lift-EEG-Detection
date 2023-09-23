import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns




def evaluate_model(model, data:np.array, labels:np.array):
    """
    Evaluate the performance of a given machine learning model on a dataset.

    Args:
        model (estimator): The machine learning model to evaluate.
        data (array-like): The input data for evaluation.
        labels (array-like): The corresponding labels for the input data.

    Returns:
        None
    """
    labels = np.ravel(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    


    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    


    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    class_labels = sorted(set(y_test))

    plt.figure(figsize=(10, 8))
    sns.barplot(x=class_labels, y=precision, color='blue', alpha=0.8, label='Precision')
    sns.barplot(x=class_labels, y=recall, color='green', alpha=0.8, label='Recall')
    sns.barplot(x=class_labels, y=f1, color='yellow', alpha=0.8, label='F1-score')

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-score for each Class")
    plt.legend()
    plt.show()



    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def grid_search_cv(model, data: np.array, labels: np.array, param_grid: dict):
    """
    Evaluate the performance of a given machine learning model on a dataset with hyperparameter tuning.

    Args:
        model (estimator): The machine learning model to evaluate.
        data (array-like): The input data for evaluation.
        labels (array-like): The corresponding labels for the input data.
        param_grid (dict): The hyperparameter grid for GridSearchCV.

    Returns:
        None
    """
    labels = np.ravel(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
     

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    

    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_params_




def evaluate_knn_classifier(data, labels, k_range=range(1, 100), test_size=0.2, random_state=42):
    """
    Evaluate a K-Nearest Neighbors (KNN) classifier for different values of k and plot the accuracy scores.

    Args:
        data (array-like): The input data for classification.
        labels (array-like): The corresponding labels for the input data.
        k_range (iterable, optional): Range of k values to evaluate. Defaults to range(1, 100).
        test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
        random_state (int, optional): Seed for random number generation. Defaults to 42.

    Returns:
        None
    """

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)


    accuracy_scores = []

    # Loop through different k values
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)


    plt.figure(figsize=(8, 6))
    plt.plot(k_range, accuracy_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Neighbors (k)')
    plt.grid(True)
    plt.show()


def create_resnet(input_shape,num_classes=7):
    """
    Creates a ResNet model for classification.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (224, 224, 3) for RGB images).
        num_classes (int, optional): Number of output classes. Defaults to 7.

    Returns:
        tf.keras.Model: A ResNet model for classification.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution layer
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(4):
        x = residual_block(x, filters=64)


    x = layers.GlobalAveragePooling1D()(x)


    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)  



    model = models.Model(inputs, outputs)
    return model


def residual_block(x, filters, kernel_size=3, stride=1):

    """
    Creates a residual block for a ResNet model.

    Args:
        x (tf.Tensor): Input tensor to the residual block.
        filters (int): Number of filters in the convolutional layers.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride for the convolutional layers. Defaults to 1.

    Returns:
        tf.Tensor: Output tensor of the residual block.
    """
    shortcut = x

    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)


    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x