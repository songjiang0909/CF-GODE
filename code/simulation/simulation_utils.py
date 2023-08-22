import numpy as np
import scipy.io as sio
import pickle as pkl
from utils import *


def read_data(dataset):
    
    if dataset == "BlogCatalog":
        data = sio.loadmat("../data/BlogCatalog/BlogCatalog_processed.mat")
        with open('../data/BlogCatalog/BlogCatalog_parts.pkl','rb') as f:
            parts = pkl.load(f)
    
    if dataset == "Flickr":
        data = sio.loadmat("../data/Flickr/Flickr_processed.mat")
        with open('../data/Flickr/Flickr_parts.pkl','rb') as f:
            parts = pkl.load(f)


    return data,parts



def data_split(parts):

    train_index = []
    val_index = []
    test_index = []
    for i in range(len(parts["membership"])):
        if parts["membership"][i]==0:
            train_index.append(i)
        elif parts["membership"][i]==1:
            val_index.append(i)
        else:
            test_index.append(i)
    print ("Size of train graph:{}, val graph:{}, test graph:{}".format(len(train_index),len(val_index),len(test_index)))


    return train_index,val_index,test_index



def adj_split(data,train_index,val_index,test_index,dataset):

    A = data["Network"]
    train_A = np.array([a[train_index] for a in A[train_index]])
    val_A = np.array([a[val_index] for a in A[val_index]])
    test_A = np.array([a[test_index] for a in A[test_index]])
    print ("Shape of adj matrix train:{}, val:{}, test:{}".format(train_A.shape,val_A.shape,test_A.shape))


    return train_A,val_A,test_A



def save_sim(args,dataset,out):
    
    setting = create_eid_sim(args)
    if dataset == "BlogCatalog":
        file = f"../data/BlogCatalog/{setting}_simulated.pkl"

    if dataset == "Flickr":
        file = f"../data/Flickr/{setting}_simulated.pkl"

    with open(file,'wb') as f:
        pkl.dump(out,f)
    print ("save data to : {}".format(file))

    return 



def load_sim(args,dataset):


    setting = create_eid_sim(args)
    if dataset == "BlogCatalog":
        file = f"../data/BlogCatalog/{setting}_simulated.pkl"

    if dataset == "Flickr":
        file = f"../data/Flickr/{setting}_simulated.pkl"
    
    with open(file,'rb') as f:
        data = pkl.load(f)

    return  data



# def load_sim_debug(dataset,norm=False):
    
#     if dataset == "BlogCatalog":
#         if not norm:
#             file = "../../data/BlogCatalog/BlogCatalog_simulated.pkl"
#         else:
#             file = "../../data/BlogCatalog/BlogCatalog_simulated_norm.pkl"
#     if dataset == "Flickr":
#         if not norm:
#             file = "../../data/Flickr/Flickr_simluated.pkl"
#         else:
#             file = "../../data/Flickr/Flickr_simluated_norm.pkl"
    
#     with open(file,'rb') as f:
#         data = pkl.load(f)

#     return  data


# def read_data_debug(dataset):
    
#     if dataset == "BlogCatalog":
#         data = sio.loadmat("../../data/BlogCatalog/BlogCatalog_processed.mat")
#         with open('../../data/BlogCatalog/BlogCatalog_parts.pkl','rb') as f:
#             parts = pkl.load(f)
    
#     if dataset == "Flickr":
#         data = sio.loadmat("../../data/Flickr/Flickr_processed.mat")
#         with open('../../data/Flickr/Flickr_parts.pkl','rb') as f:
#             parts = pkl.load(f)


#     return data,parts


# def save_sim_debug(dataset,out):
    
#     if dataset == "BlogCatalog":
#         file = "../../data/BlogCatalog/BlogCatalog_simulated.pkl"

#     if dataset == "Flickr":
#         file = "../../data/Flickr/Flickr_simluated.pkl"
    
#     with open(file,'wb') as f:
#         pkl.dump(out,f)
#     print ("save data to : {}".format(file))

#     return