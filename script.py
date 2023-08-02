import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import glob
import matplotlib as plt
from collections import OrderedDict
import random
from scripts_spyder_eegsignals import *
# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os
from sklearn.model_selection import train_test_split


###importing all of subj1 data and events
path_data = "C:\\Users\\amoha\\Downloads\\train\\subj1_data"
path_label= "C:\\Users\\amoha\\Downloads\\train\\subj1_events"
subj1_data, subj1_label = [] , []
for file in glob.glob(path_data + "\\*.csv"):
    subj1_data.append(file)
for file in glob.glob(path_label + "\\*.csv"):
    subj1_label.append(file)

all_data = pd.DataFrame()
all_labels = pd.DataFrame()
for i,j in zip(subj1_data, subj1_label):
    dataa = pd.read_csv(f"{i}")
    events = pd.read_csv(f"{j}")
    dataa.drop(["id"],axis = 1, inplace = True)
    events.drop(["id"],axis = 1, inplace = True)
    all_data = all_data.append(dataa)
    all_labels = all_labels.append(events)
### making an array of data were at least 1 class is occuring
start_end_data = start_end_data_finder(all_labels)
data_extracted_occurances = np.reshape(data_extractor(start_end_data,all_data),(6*260,149,32) )

### making an array containing data were no class is occuring
no_events_data_extracted = data_extractor_noevent(all_data, all_labels,1560)
final_data = np.empty(0*149*32)
final_data = np.concatenate((data_extracted_occurances,no_events_data_extracted))
final_data = np.reshape(final_data, (3120,149*32))

class_labels = np.ones(1560)

for i in range(6):
    class_labels[i*260:(i+1)*260] = class_labels[i*260:(i+1)*260] *(i)

noevent_label = np.ones(1560)*(6)

all_class_labels = np.concatenate((class_labels,noevent_label))


###########Scaling and Shuffling all data and labels:##############
from sklearn.preprocessing import StandardScaler
all_data_shuffled , all_labels_shuffled = shuffle(final_data, class_labels, random_state = 0)


ss = StandardScaler()
all_data_shuffled_scaled = ss.fit_transform(all_data_shuffled)

##########Principal Component Analysis(PCA):###########
    
    
    
from sklearn.decomposition import PCA
    
n_components = 15
pca = PCA(n_components = n_components)
pca.fit(all_data_shuffled)
pca_tr = pca.fit_transform(all_data_shuffled_scaled)



# Naming the PCs by iteration
# pc_cols = [f'PC{n}' for n in range(1, n_components + 1)]
# # Creating a pd.DataFrame containing the explained variance ratio from the PCA
# pc_var_df = pd.DataFrame(
#     {
#         'Variance': pca.explained_variance_ratio_,
#         'PC': pc_cols
#     }
# )

# # Creating a subplot of the cumulative variance ratio & Eigenvalues
# fig, ax1 = plt.subplots(figsize=(25, 10))
# ax2 = ax1.twinx()

# # Creating a barplot
# sns.barplot(
#     x='PC', y='Variance',
#     data=pc_var_df,
#     label='PC',
#     color='tab:red',
#     ax=ax1
# )

# ax1.set_ylabel('Explained Variance (Eigenvalues)')
# # Plotting the cumulative sum of Explained Variance Ratio
# ax2.plot(
#     np.cumsum(pca.explained_variance_ratio_),
#     label='Cumulative Explained Variance Ratio'
# )
# plt.title('Scree Plot')
# ax2.set_ylabel('Cumulative Explained Variance Ratio')
# ax1.legend(loc=2)
# ax2.legend()
# ax2.grid(b=None)
# plt.show()
# pc_var_df

#### dividing the data into train and test:
X_train, X_test, y_train, y_test = train_test_split(all_data_shuffled_scaled, 
                                                    all_labels_shuffled, 
                                                    test_size=0.2, random_state=1)    


###########implementing XGboost:###########
    
    
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
kfold_validation = KFold(5)
X = all_data_shuffled_scaled
y = all_labels_shuffled

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
XGBoost_acc = accuracy_score(y_test, y_pred)

mod_score=cross_val_score(model,X,y,cv=kfold_validation)


print(f"The accuracy of this XGBoost  is {XGBoost_acc}") , print(mod_score)



##########Support Vector machines###########


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

clf = SVC(kernel ="rbf" )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
SVM_acc = accuracy_score(y_test,y_pred)





###########Random Forest Model###########
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=1000)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
RandomForest_acc = metrics.accuracy_score(y_test, y_pred)

#################Implementing K nearest neighbor##############
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn5 = KNeighborsClassifier(n_neighbors = 5)
knn1 = KNeighborsClassifier(n_neighbors=1)
knn10 = KNeighborsClassifier(n_neighbors=10)

knn5.fit(X_train, y_train)
knn1.fit(X_train, y_train)
knn10.fit(X_train, y_train)

y_pred_10 = knn10.predict(X_test)
y_pred_5 = knn5.predict(X_test)
y_pred_1 = knn1.predict(X_test)

print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)
print("Accuracy with k=10", accuracy_score(y_test, y_pred_10)*100)
