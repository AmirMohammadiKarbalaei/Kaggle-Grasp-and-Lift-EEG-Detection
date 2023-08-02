# Kaggle-Grasp-and-Lift-EEG-Detection: Empowering Patients with Neurological Disabilities

This project is dedicated to revolutionizing the lives of patients with neurological disabilities and amputations by enabling them to regain the ability to perform basic activities. To achieve this, we have undertaken the challenge of identifying simple hand functions through cutting-edge EEG detection.


The Training and Test data essential for this project can be accessed through the Kaggle Grasp-and-Lift EEG Detection competition. :https://www.kaggle.com/c/grasp-and-lift-eeg-detection


Data preprocessing involved meticulous cleaning and division into sections, each comprising 150 rows of sensor data. By converting data and labels into arrays, we've significantly improved computational efficiency. Employing a standard scaler and assessing the suitability of PCA, we found that PCA with 25 components yielded comparable accuracy to using all data components.


Different algorithms such as SVM, KNN, XGboost and Random Forest have been implemented on the data. with Random Forest having the highest accuracy of above 92%.
