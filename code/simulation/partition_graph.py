import numpy as np
import pymetis
import scipy.io as sio
import pickle as pkl
import os.path as osp
from datatools import *


# dataset = "Flickr"
dataset = "Flickr"


# data = sio.loadmat('../../data/'+Flickr'/Flickr0.mat')
data = sio.loadmat(osp.join("..","..","data",dataset,dataset+"0.mat"))
A = data["Network"].toarray()
adjacency_list = [np.argwhere(A[i]==1).flatten() for i in range(A.shape[0])]
n_cuts, membership = pymetis.part_graph(3, adjacency=adjacency_list)

parts_save = {
        'n_cuts':n_cuts,
        'membership':membership
        }

train_index,val_index,test_index = data_split(parts_save)


A = data["Network"].toarray()
X = data["Attributes"]
print ("X",X.shape)



train_A = np.array([a[train_index] for a in A[train_index]])
val_A = np.array([a[val_index] for a in A[val_index]])
test_A = np.array([a[test_index] for a in A[test_index]])
print ("Shape of adj matrix train:{}, val:{}, test:{}".format(train_A.shape,val_A.shape,test_A.shape))

sum_trA = np.sum(train_A,1)
sum_vlA = np.sum(val_A,1)
sum_ttA = np.sum(test_A,1)
print (np.sum(sum_trA==0)+np.sum(sum_vlA==0)+np.sum(sum_ttA==0))
print (np.sum(sum_trA)+np.sum(sum_vlA)+np.sum(sum_ttA))


to_remove =[]
for i in range(len(sum_trA)):
    if sum_trA[i]==0:
        to_remove.append(train_index[i])
for i in range(len(sum_vlA)):
    if sum_vlA[i]==0:
        to_remove.append(val_index[i])
for i in range(len(sum_ttA)):
    if sum_ttA[i]==0:
        to_remove.append(test_index[i])
print ("To remove:")
print(len(to_remove))


to_remove = set(to_remove)
num,col = A.shape
print (A.shape)


new_Graphid = []
for i in range((num)):
    if i not in to_remove:
        new_Graphid.append(i)
print (len(new_Graphid))
print(len(set(new_Graphid) | set(to_remove)))
# print(trainIndex)

newX = X[new_Graphid]
newA = np.array([a[new_Graphid] for a in A[new_Graphid]])
print (newX.shape)
print (newA.shape)

sio.savemat(osp.join("..","..","data",dataset,dataset+"_processed.mat"),{'Attributes': newX,'Network':newA})

new_parts = []
for i in range(len(membership)):
    if i not in to_remove:
        new_parts.append(membership[i])
print (len(new_parts))
print (new_parts[-15:])
print (np.array(membership)[new_Graphid][-15:])


parts_save = {
        'n_cuts':n_cuts,
        'membership':new_parts
        }

with open(osp.join("..","..","data",dataset,dataset+"_parts.pkl"),'wb') as f:
    pkl.dump(parts_save,f)






