import numpy as np
import pandas as pd 
from scripts_spyder_eegsignals import *
subj2_data = pd.read_csv("subj2_series1_data.csv")
subj2_labels = pd.read_csv("subj2_series1_events.csv")
test_data_labels = subj2_labels.drop("id", axis = 1)
test_data = subj2_data.drop("id", axis = 1)


start_end_test_data = start_end_data_finder(test_data_labels)
        
test_data_extracted = data_extractor(start_end_test_data, test_data)
test_data_extracted = np.reshape(test_data_extracted,(6*28,149,32))
test_data_noevent_extracted = data_extractor_noevent(test_data,test_data_labels,6*28)

test_data_all = np.concatenate((test_data_extracted,test_data_noevent_extracted))
test_data_all = np.reshape(test_data_all,(336,149*32))
test_data_ylabel = np.ones((336))
    
for i in range(6):
    test_data_ylabel[i*28:(i+1)*28] = test_data_ylabel[i*28:(i+1)*28] *(i)
test_data_ylabel[6*28:] = test_data_ylabel[6*28:] *(6)



from sklearn.preprocessing import StandardScaler
all_testdata_shuffled , all_testlabels_shuffled = shuffle(test_data_all, test_data_ylabel, random_state = 0)


ss = StandardScaler()
all_testdata_shuffled_scaled = ss.fit_transform(all_testdata_shuffled)

from sklearn.decomposition import PCA
    
n_components = 20
pca = PCA(n_components = n_components)
pca.fit(all_data_shuffled_scaled)
pca_tr_test = pca.fit_transform(all_data_shuffled_scaled)
    