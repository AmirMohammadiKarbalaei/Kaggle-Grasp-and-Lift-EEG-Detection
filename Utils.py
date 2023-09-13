import pandas as pd
import numpy as np
import random
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import glob
import os
from tqdm import tqdm

def evaluate_model(model, data, labels):
    """
    This function evaluates the performance of a given machine learning model
      on a dataset. It splits the data into training and testing sets, trains 
      the model, and predicts on the test set. The function calculates and displays
        the accuracy and confusion matrix of the predictions. It also computes the 
        precision, recall, and F1-score for each class and visualizes them using a 
        bar plot. Finally, it prints a classification report summarizing the 
        model's performance for each class.
    
    """
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


def data_extractor(start_end_data:pd.DataFrame,data1:pd.DataFrame):
    """
    Extracts specific rows of data based on event 
    start and end indexes and returns a NumPy array 
    with the extracted data.
    
    
    
    """
    data = []
    for i in start_end_data.columns:
        
        rows = []
        for j in range(0, len(start_end_data[f"{i}"]), 2):
            start_idx = start_end_data[f"{i}"][j]
            end_idx = start_end_data[f"{i}"][j+1]
            rows.append(np.array(data1.iloc[start_idx:end_idx+1]))

        data.append(np.array(rows))

    #datas = np.reshape(datas,(6*260,149,32))
    return np.array(data)





def start_end_data_finder(events:pd.DataFrame):
    """

    Finds start and end row indexes for each event 
    in the data and returns them in a DataFrame.

    """
    dic = {}
    for i in events.columns:
        ones = []
        for idx,j in enumerate(events[i]):
            if j == 1:
                ones.append(idx)
            dic[i] = ones
    start_end = {}    
    for idx,event in enumerate(dic.keys()):
        start_end_vals =[dic[f"{event}"][0]]
        for t,i in enumerate(dic[f"{event}"]):
            if t == len(dic[f"{event}"])-1:
                break
            if dic[f"{event}"][t+1] - dic[f"{event}"][t] !=1:
                start_end_vals.append(dic[f"{event}"][t])
                start_end_vals.append(dic[f"{event}"][t+1])
        start_end_vals.append(dic[f"{event}"][len(dic[f"{event}"])-1])
        start_end[f"{event}"] = start_end_vals 
    #start_end_data = pd.DataFrame.from_dict(start_end)
    return start_end





def random_indexes_noevent(event:pd.DataFrame):
    """
    Generates random row indexes with no events occurring 
    and returns them as a list.
    
    """
    indexes = []
    for i in range(1000):
        rand = random.choice(range(1000))
        num = np.sum(event.iloc[rand:rand+149])
        if num.any() == False:
            indexes.append(rand)
    return indexes




def data_extractor_noevent(data:pd.DataFrame,event:pd.DataFrame,number_of_consecutive_rows:int):
    """
    
    Extracts random sets of consecutive rows with no events 
    from the data and returns them as a NumPy array.
    
    """
    events_rows = []
    indexes = random_indexes_noevent(event)
    for i in range(number_of_consecutive_rows):
        randy = random.choice(indexes)
        events_rows.append(np.array(data.iloc[randy:randy+150]))
    events_rows = np.array(events_rows)
    return events_rows




def load_and_save_data(subject_count=8, data_path_template="C:\\Users\\amoha\\Downloads\\train\\subj{}_data",
                       label_path_template="C:\\Users\\amoha\\Downloads\\train\\subj{}_events"):
    all_data_list = []
    all_labels_list = []

    for subject_id in range(1, subject_count + 1):
        subj_data = []
        subj_labels = []
        data_path = data_path_template.format(subject_id)
        label_path = label_path_template.format(subject_id)

        data_files = glob.glob(data_path + "\\*.csv")
        label_files = glob.glob(label_path + "\\*.csv")

        # Use tqdm to create a progress bar
        for data_file, label_file in tqdm(zip(data_files, label_files), total=len(data_files), desc=f"Subject {subject_id}"):
            data = pd.read_csv(data_file)
            labels = pd.read_csv(label_file)
            data.drop(["id"], axis=1, inplace=True)
            labels.drop(["id"], axis=1, inplace=True)
            subj_data.append(data)
            subj_labels.append(labels)

        # Concatenate data and labels for the current subject
        all_data = pd.concat(subj_data, ignore_index=True)
        all_labels = pd.concat(subj_labels, ignore_index=True)

        all_data_list.append(all_data)
        all_labels_list.append(all_labels)

    return all_data_list, all_labels_list
