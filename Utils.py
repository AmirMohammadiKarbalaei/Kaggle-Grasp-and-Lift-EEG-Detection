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
    for i in start_end_data.keys():
        
        rows = []
        for j in range(0, len(start_end_data[f"{i}"]), 2):
            start_idx = start_end_data[f"{i}"][j]
            end_idx = start_end_data[f"{i}"][j+1]
            rows.append(np.array(data1.iloc[start_idx:end_idx+1]))

        data.append(rows)
    return data




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




def random_indexes_noevent(event: pd.DataFrame, max_rows: int = 1500, threshold = 20):
    """
    Generates random row indexes with no events occurring
    and returns them as a list.

    Args:
        event (pd.DataFrame): DataFrame containing event data.
        max_rows (int): Maximum number of rows to consider. Default is 1500.

    Returns:
        list: List of random row indexes with no events.
    """
    indexes = []
    for i in range(max_rows):
        rand = random.choice(range(max_rows - threshold+1))  
        num = np.sum(event.iloc[rand:rand+51].sum(axis=1))
        if num == 0:
            indexes.append(rand)
    return indexes




def data_extractor_noevent(data, event, number_of_consecutive_rows,threshold):
    """
    Extracts random sets of consecutive rows with no events
    from the data and returns them as a NumPy array.

    Args:
        data (pd.DataFrame): DataFrame containing data.
        event (pd.DataFrame): DataFrame containing events.
        number_of_consecutive_rows (int): Number of consecutive rows to extract.

    Returns:
        np.ndarray: NumPy array containing the extracted rows.
    """
    events_rows = []
    indexes = random_indexes_noevent(event,threshold = threshold)
    for i in range(number_of_consecutive_rows):
        randy = random.choice(indexes)
        events_rows.append(data.iloc[randy:randy+threshold+1].values)  # Use .values to get NumPy array
    events_rows = np.array(events_rows)
    return events_rows





import pandas as pd
import glob
from tqdm import tqdm
import logging

DATA_PATH_TEMPLATE = "C:\\Users\\amoha\\Downloads\\train\\subj{}_data"
LABEL_PATH_TEMPLATE = "C:\\Users\\amoha\\Downloads\\train\\subj{}_events"

def load_data(subject_count: int = 8, data_path_template: str = DATA_PATH_TEMPLATE,
              label_path_template: str = LABEL_PATH_TEMPLATE) -> tuple:
    """
    Load data and labels for multiple subjects.

    Args:
        subject_count (int): The number of subjects to load.
        data_path_template (str): The template for the data file paths.
        label_path_template (str): The template for the label file paths.

    Returns:
        Tuple: Two lists containing concatenated data and labels DataFrames for each subject.
    """
    all_data_list = []
    all_labels_list = []

    for subject_id in range(1, subject_count + 1):
        subject_data_list = []  # List to store data DataFrames for the current subject
        subject_labels_list = []
        data_path = data_path_template.format(subject_id)
        label_path = label_path_template.format(subject_id)

        data_files = glob.glob(data_path + "\\*.csv")
        label_files = glob.glob(label_path + "\\*.csv")

        if not data_files or not label_files:
            logging.warning(f"No files found for Subject {subject_id}. Skipping.")
            continue

        for data_file, label_file in tqdm(zip(data_files, label_files), total=len(data_files), desc=f"Subject {subject_id}"):
            try:
                data = pd.read_csv(data_file)
                labels = pd.read_csv(label_file)
                data.drop(["id"], axis=1, inplace=True)
                labels.drop(["id"], axis=1, inplace=True)
                subject_data_list.append(data)
                subject_labels_list.append(labels)
            except Exception as e:
                logging.error(f"Error loading data for Subject {subject_id}: {str(e)}")

        if subject_data_list and subject_labels_list:
            all_data = pd.concat(subject_data_list, ignore_index=True)
            all_labels = pd.concat(subject_labels_list, ignore_index=True)
            all_data_list.append(all_data)
            all_labels_list.append(all_labels)

    return all_data_list, all_labels_list


def process_start_end(start_end:list, threshold:int):
    """
    Process the start_end data by applying a threshold to the values.

    Args:
        start_end (list): A list of dictionaries containing start and end values.
        threshold (int): The threshold value to compare against. Values greater
            than the threshold will be adjusted.

    Returns:
        list: The processed start_end data with adjusted values.
    """
    for i in range(len(start_end)):
        for key, value in start_end[i].items():
            new_value = []
            found_pair = False
            for idx in range(0, len(value), 2):

                a = value[idx + 1] - value[idx]
                if a > threshold:
                    x = value[idx]
                    y = x + threshold
                    new_value.extend([x, y])
                    found_pair = True
                elif a == threshold:
                    new_value.extend([value[idx], value[idx + 1]])
                elif a < threshold:
                    # If the difference is less than the threshold, remove these values
                    value[idx] = None
                    value[idx + 1] = None

            # Filter out None values (values less than the threshold) from the list
            new_value = [v for v in new_value if v is not None]
            start_end[i][key] = new_value
    
    return start_end



def process_load_labels(load_labels,number_of_subj):
    """
    Process a list of dataframes(labels) by filtering rows where the sum of values is less than or equal to 1.
    This function removes data points where there are more than one class present.

    Args:
        load_labels (list of DataFrame): A list of DataFrames to be processed.

    Returns:
        list of DataFrame: The processed list of DataFrames.
    """
    processed_load_labels = []
    
    for i in range(number_of_subj):
        df_copy = load_labels[i].copy()
        df_copy['sum'] = df_copy.sum(axis=1)
        df_copy = df_copy[df_copy['sum'] <= 1]
        df_copy = df_copy.drop(columns=['sum'])
        processed_load_labels.append(df_copy)
    
    return processed_load_labels

def calculate_class_data(all_extracted_data):
    class_data = {class_idx: [] for class_idx in range(6)}
    summed_data = [[] for _ in range(6)]

    for sublist in all_extracted_data:
        for class_index, class_data_list in enumerate(sublist):
            class_data[class_index].extend(map(len, class_data_list))
            summed_data[class_index].extend(class_data_list)

    return class_data, summed_data


def calculate_combined_dict(class_data):
    combined_dict = {class_idx: len(lengths) for class_idx, lengths in class_data.items()}
    return combined_dict

def print_class_summary(combined_dict):
    for class_idx, total_length in combined_dict.items():
        print(f"Class {class_idx}: Data points = {total_length}")

def calculate_min_length(summed_data):
    min_length = np.min([len(inner_array) for inner_array in summed_data])
    return min_length



def apply_data_augmentation(eeg_signal):
    # Randomly select a data augmentation technique
    augmentation_technique = np.random.choice(['time_warp', 'noise_injection', 'signal_shift'])

    augmented_signal = eeg_signal.copy()

    for technique in augmentation_technique:
        if technique == 'time_warp':
            # Apply time warping by scaling the time axis
            scaling_factor = np.random.uniform(0.9, 1.1)
            augmented_signal = np.interp(np.arange(len(augmented_signal)) * scaling_factor, np.arange(len(augmented_signal)), augmented_signal)

        elif technique == 'noise_injection':
            # Add random noise to the signal and cast to 'int64'
            noise_level = np.random.uniform(0.01, 0.1)
            noise = np.random.normal(0, noise_level, size=len(augmented_signal)).astype('int64')
            augmented_signal += noise

        elif technique == 'signal_shift':
            # Shift the signal in time
            shift_amount = np.random.randint(-50, 50)  # Adjust the range as needed
            augmented_signal = np.roll(augmented_signal, shift_amount)

    return augmented_signal
