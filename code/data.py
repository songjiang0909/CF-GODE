import torch
import scipy.sparse as sp
import pickle as pkl
# from simulation.ds_simulation import *
from simulation.ds_simulation import *
from simulation.simulation_utils import *


def data_prepare(args):

    if args.simulate:
        train_outputs, val_outputs, test_outputs = data_simulation(args,dataset=args.dataset,norm=args.norm)
        outputs = {"train_outputs":train_outputs, "val_outputs":val_outputs, "test_outputs":test_outputs}
        save_sim(args,args.dataset,outputs)
    else:
        outputs = load_sim(args,args.dataset)

        
    
    print ("Got data!")

    if args.norm:
        mean ,std = outputs["train_outputs"]["mean"]["health_condition"],outputs["train_outputs"]["std"]["health_condition"]
        train_outputs = data_normalzation(outputs["train_outputs"],mean ,std)
        val_outputs = data_normalzation(outputs["val_outputs"],mean ,std)
        test_outputs = data_normalzation(outputs["test_outputs"],mean ,std)
        outputs = {"train_outputs":train_outputs, "val_outputs":val_outputs, "test_outputs":test_outputs}
        print ("Data normalized!")
        
    return outputs




def data_normalzation(data,mean ,std):

    features = ["health_condition","cf_health_condition"]
    for k in features:
        data[k+"_norm"] = (data[k] - mean) / std


    return data



def numpy_to_torch(args,data):

    FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    keys = ['adj','health_condition', 'action_application_point','avg_neighbor_action_application', 'health_condition_norm','cf_health_condition', 'cf_action_application_point','cf_avg_neighbor_action_application', 'cf_health_condition_norm']
    for k in keys:
        if k in set(['adj']):
            adj = data[k]
            adj = adj + np.multiply(adj.T,(adj.T > adj)) - np.multiply(adj,adj.T > adj)
            adj+=np.identity(adj.shape[0],dtype=int)
            adj = adj.astype(float)
            adj = normalize(adj)
            data[k] = FloatTensor(adj)
        else:
            data[k] = FloatTensor(np.expand_dims(data[k], axis=2))
    
    data["feature"] = FloatTensor(data["feature"])
    feature_aug = [data["feature"] for _ in range(args.num_time_steps)]
    data["feature"] = torch.stack(feature_aug,1)
    print ("=="*30)
    print (data["feature"].shape)

    return data



def preprocessing(args,outputs):
    
    outputs["train_outputs"] = numpy_to_torch(args,outputs["train_outputs"])
    outputs["val_outputs"] = numpy_to_torch(args,outputs["val_outputs"])
    outputs["test_outputs"] = numpy_to_torch(args,outputs["test_outputs"])

    return outputs



def normalize(mx):
    """
    Row-normalize sparse matrix 
    
    code from https://github.com/tkipf/pygcn
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


