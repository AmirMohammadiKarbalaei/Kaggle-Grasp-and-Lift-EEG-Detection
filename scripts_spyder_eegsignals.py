import pandas as pd
import numpy as np
import random

#defining required functions:
def data_extractor(start_end_data,data1):
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


def start_end_data_finder(events):
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




##finding start and end row idx for events  not occuring
def random_indexes_noevent(event):
    indexes = []
    for i in range(1000):
        rand = random.choice(range(1000))
        num = np.sum(event.iloc[rand:rand+149])
        if num.any() == False:
            indexes.append(rand)
    return indexes  

def data_extractor_noevent(data,event,number):
    events_rows = []
    indexes = random_indexes_noevent(event)
    for i in range(number):
        randy = random.choice(indexes)
        events_rows.append(np.array(data.iloc[randy:randy+149]))
    events_rows = np.array(events_rows)
    return events_rows
