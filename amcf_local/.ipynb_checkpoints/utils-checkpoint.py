import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import heapq
import numpy as np

# --- dataset --#

random_seed = 0

def get_data(batch_size=256):

    path = 'data_amcf/ratings'

    data = pd.read_csv(path)
    data = data.values # convert to numpy array
    inps = data[:, 0:2].astype(int) # get user, item inputs
    tgts = data[:, 2].astype(int) # get rating targets
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    loaders = []
    for train_index, test_index in kf.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        # split and convert to tensors
        inps_train = torch.tensor(inps[train_index], dtype=torch.long)
        inps_test = torch.tensor(inps[test_index], dtype=torch.long)
        tgts_train = torch.tensor(tgts[train_index], dtype=torch.long)
        tgts_test = torch.tensor(tgts[test_index], dtype=torch.long)
        # convert to TensorDataset type
        train_set = TensorDataset(inps_train, tgts_train)
        test_set = TensorDataset(inps_test, tgts_test)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        loaders.append([train_loader, test_loader])
    
    # return all loaders for cross validation
    return loaders   

# --- item_to_genre --#

def item_to_genre(item):
    file_dir = 'data_amcf/funds'
    funds = pd.read_csv(file_dir, header=0, index_col=0)
    genre = funds.loc[item]
    return genre

def get_genre(data_size):
    file_dir = 'data_amcf/funds'
    funds = pd.read_csv(file_dir, header=0)
    items = funds.iloc[:, 0].values
    genres = funds.iloc[:, 1:].values
    return (items, genres)

# --- topK --#

def getListMaxNumIndex(num_list,topk=3):
    max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    return set(list(max_num_index)), set(list(min_num_index))

#top bot k match
def topK(a, b, k=5, m=3, num_user=63619):
    results_max = np.zeros(num_user) # 943, 6040
    results_min = np.zeros(num_user)
    for i in range(num_user): # 943
        Max1,Min1 = getListMaxNumIndex(list(a[i]),m)
        Max2,Min2 = getListMaxNumIndex(list(b[i]),k)
        results_max[i] = len(Max1&Max2)/m
        results_min[i] = len(Min1&Min2)/m
    return results_max.mean(), results_min.mean()

#hit ratio @k
def hrK(a, b, k=5, num_user=63619):
    # a = pred40
    # b = pref
    results_max = np.zeros(num_user)
    results_min = np.zeros(num_user)
    for i in range(num_user):
        Max1,Min1 = getListMaxNumIndex(list(a[i]),k)
        Max2,Min2 = getListMaxNumIndex(list(b[i]),1)
        results_max[i] = len(Max1&Max2)
        results_min[i] = len(Min1&Min2)
    return results_max.mean()
