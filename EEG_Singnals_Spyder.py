import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import glob
import matplotlib as plt
from collections import OrderedDict
import random
from scripts_spyder_eegsignals import *


###importing data 
path_data = "C:\\Users\\amoha\\Downloads\\train\\subj1_data"
path_label= "C:\\Users\\amoha\\Downloads\\train\\subj1_events"
subj1_data, subj1_label = [] , []
for file in glob.glob(path_data + "\\*.csv"):
    subj1_data.append(file)
for file in glob.glob(path_label + "\\*.csv"):
    subj1_label.append(file)

##Making a ingle data/label for all of the available data/labels for first subject
all_data = pd.DataFrame()
all_labels = pd.DataFrame()
for i,j in zip(subj1_data, subj1_label):
    dataa = pd.read_csv(f"{i}")
    events = pd.read_csv(f"{j}")
    dataa.drop(["id"],axis = 1, inplace = True)
    events.drop(["id"],axis = 1, inplace = True)
    all_data = all_data.append(dataa)
    all_labels = all_labels.append(events)
###creating an array consisting of rows of the data relating to occurance of each of the classes
start_end_data = start_end_data_finder(all_labels)
data_extracted_occurances = np.reshape(data_extractor(start_end_data,all_data),(6*260,149,32) )
###creating an array consisting of rows of the data were none of classes are occuring
no_events_data_extracted = data_extractor_noevent(all_data, all_labels)
random_val_found_noevent = random_indexes_noevent(all_labels)

final_data = np.empty(0*149*32)
final_data = np.concatenate((data_extracted_occurances,no_events_data_extracted))

### creating an array for the labels of the data
class_labels = np.ones(1820)
for i in range(7):
    class_labels[i*260:(i+1)*260] = class_labels[i*260:(i+1)*260] *(i)
final_data = np.reshape(final_data, (1820,149*32))
##shuffling all data and labels:
    
all_data_shuffled , all_labels_shuffled = shuffle(final_data, class_labels, random_state = 0)
    
###implementing XGboost:
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
kfold_validation = KFold(15)
X = all_data_shuffled
y = all_labels_shuffled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=np.arange(0.3, 1, 0.1),
              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
              gamma=[0.01, 0.1, 0.3, 0.5, 1, 1.5, 2], gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=[0.0001, 0.001, 0.01, 0.1],
              max_delta_step=0, max_depth=range(2, 10), min_child_weight=1, missing=0,
              monotone_constraints='()', n_estimators=range(50, 400, 10), n_jobs=16,
              num_parallel_tree=1, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=[1, 1.5, 2, 3, 4.5], scale_pos_weight=None, subsample=[0.8, 0.85, 0.9, 1.0],
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1, verbosity=None)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

mod_score=cross_val_score(model,X,y,cv=kfold_validation)


print(f"The accuracy of this model is {accuracy}") , print(mod_score)
