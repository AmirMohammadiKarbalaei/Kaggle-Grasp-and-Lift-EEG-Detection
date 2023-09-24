# Project Title: Neurological Disability Hand Function Identification

### Introduction

This project aims to address the critical need for assisting patients with neurological disabilities or limb amputations in regaining the ability to perform fundamental hand functions. To achieve this, we leverage EEG (Electroencephalogram) data analysis to identify and understand simple hand movements.

### Data Source

The dataset used for this project is sourced from the Kaggle Grasp-and-Lift EEG Detection competition, accessible at [Kaggle Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection). It includes EEG recordings collected from 8 subjects.

### Data Preprocessing

The initial dataset underwent a thorough cleaning process, resulting in organized sections, each comprising 21 rows of sensor data. To expedite computational efficiency, both data and corresponding labels were transformed into arrays. Further, data normalization was performed using a standard scaler. The feasibility of Principal Component Analysis (PCA) was explored, demonstrating that PCA with 60 components achieved comparable accuracy to using the entire dataset.

### Model Implementation

Various machine learning algorithms, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), XGBoost, and Random Forest, were applied to the preprocessed data. The highest achieved accuracy among these models was approximately 32%. Notably, this performance is significantly better than random guessing, considering the presence of 7 distinct classes in the dataset.

Additionally, Convolutional Neural Networks (CNN) and Residual Networks (ResNet) were implemented to delve deeper into EEG signal analysis. Surprisingly, these advanced models yielded accuracy levels similar to the simpler models, indicating that the current EEG signals may not provide sufficient information for a more complex understanding of underlying patterns.

### Future Directions

To enhance the accuracy of the models and gain a deeper understanding of the EEG signals, several avenues for future work are proposed. Feature engineering is a crucial next step to extract more relevant information from the data. Furthermore, the treatment of EEG signals as raw data can be revised to incorporate a temporal unit into the code, potentially leading to increased accuracy and improved insights into neurological disability-related hand movements.
