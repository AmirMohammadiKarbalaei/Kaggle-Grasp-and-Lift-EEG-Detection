# Kaggle-Grasp-and-Lift-EEG-Detection

Identifying simple hand functions to help patients with neurological disabilities/amputation to gain the ability to perform these basic activities.


Training and Test data can be downloaded from Kaggle Grasp-and-Lift EEG Detection competition:https://www.kaggle.com/c/grasp-and-lift-eeg-detection

data for subj1 has been used to train the model. A scripts file was made containing all of the required functions to increase readability of the main body of code. 
data has first been cleaned and devided into sections whith  each section containing 149 rows of data from sensors. data and labels have been made into array to increase computaion time. all data has been scaled using a standard scaler and suitability of PCA has also been checked. PCA with 25 components has proved to have similar accuracy to using the all of the data components

***(In this model data has been divided to  segments each containing 149 rows of data for occurance of classes as well as nothin occuring. this means each predicion has 149*32 components which as mentioned can be reduced to 25 to increase computational greatly)***

Different algorithms such as SVM, KNN, XGboost and Random Forest have been implemented on the data. with XGboost having the highest accuracy of above 77%.

once the model was trained, data from subj2 was used to chech the accuracy of the mmodel predcitions on new data.
