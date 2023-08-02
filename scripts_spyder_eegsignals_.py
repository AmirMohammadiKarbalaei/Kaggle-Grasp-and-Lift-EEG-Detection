import pandas as pd
import numpy as np
import random
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report


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
    sns.barplot(x=class_labels, y=precision, color='blue', alpha=1, label='Precision')
    sns.barplot(x=class_labels, y=recall, color='green', alpha=1, label='Recall')
    sns.barplot(x=class_labels, y=f1, color='yellow', alpha=1, label='F1-score')

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
            rows.append(np.array(data1.iloc[start_idx:end_idx]))

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
    df = pd.DataFrame.from_dict(dic)
    start_end = {}    
    for idx,event in enumerate(df.columns):
        start_end_vals =[df[f"{event}"][0]]
        for t,i in enumerate(df[f"{event}"]):
            if t == len(df[f"{event}"])-1:
                break
            if df[f"{event}"][t+1] - df[f"{event}"][t] !=1:
                start_end_vals.append(df[f"{event}"][t])
                start_end_vals.append(df[f"{event}"][t+1])
        start_end_vals.append(df[f"{event}"][len(df[f"{event}"])-1])
        start_end[f"{event}"] = start_end_vals 
    start_end_data = pd.DataFrame.from_dict(start_end)
    return start_end_data





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
        events_rows.append(np.array(data.iloc[randy:randy+149]))
    events_rows = np.array(events_rows)
    return events_rows
