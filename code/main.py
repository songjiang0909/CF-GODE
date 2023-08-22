import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from data import *
from model import *
from exp import *
from utils import *



parser = argparse.ArgumentParser()  

##data 
parser.add_argument('--dataset', type=str, default='Flickr', help='dataset') 
parser.add_argument('--exp_id', type=int, default=0, help='Experiments id') 
parser.add_argument('--simulate', type=bool, default=True, help='simulate data T or load existing data F')
parser.add_argument('--dimension', type=int, default=10, help='Dimension reduction of the sparse node features.')
parser.add_argument('--flip_rate', type=float, default=1, help='Treatment rate of change for counterfactual prediction')
parser.add_argument('--gamma_a', type=float, default=10, help='Time-dependent confounding degree to treatment')
parser.add_argument('--gamma_n', type=float, default=10, help='Neighbor Time-dependent confounding degree to treatment')
parser.add_argument('--gamma_f', type=float, default=10, help='Fixed confounding degree to treatment')
parser.add_argument('--per_neighbor', type=float, default=3.0, help='Percentage of neighbor fixed confounders in fixed confounding degree')
parser.add_argument('--norm', type=bool, default=True, help='data normalization')
parser.add_argument('--num_time_steps', type=int, default=15, help='normalization seq length')
parser.add_argument('--observed_steps', type=int, default=10, help='normalization seq length')


## Device
parser.add_argument('--cuda', type=bool, default=True, help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu')

## Model
parser.add_argument('--hidden_dim', type=int, default=64, help='latent trajectory dimension')
parser.add_argument('--alpha_a', type=float, default=0.5, help='coeff of balancing a')
parser.add_argument('--alpha_s', type=float, default=0.5, help='coeff of balancing s')
#ODE Solver
parser.add_argument('--ode_method', type=str, default="euler", help='dopri5,rk4,euler,dopri8')
parser.add_argument('--odeint_rtol', type=float, default=1e-3, help='odeint_rtol')
parser.add_argument('--odeint_atol', type=float, default=1e-4, help='odeint_atol')


## Training
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=5000, help='training epochs')
parser.add_argument('--K', type=int, default=4, help='all/prediction loss ratio')

## Baselines
parser.add_argument('--model', type=str, default="cde", help='baselines')#["cfgode","cde","crn"]


start_time = time.time()
print ("Start time at : ")
print_time()


args = parser.parse_args()
args.cuda = True if torch.cuda.is_available() and args.cuda else False
if args.cuda:
    torch.cuda.set_device(args.gpu_id)

print (args)


simulated_data = data_prepare(args)
processed_data = preprocessing(args,simulated_data)
# print (processed_data["train_outputs"]["health_condition"].shape)
# print (processed_data["train_outputs"]["action_application_point"].shape)
# print (processed_data["train_outputs"]["avg_neighbor_action_application"].shape)
# print (processed_data.keys())
# print (processed_data["train_outputs"].keys())


model = CGODE(args=args,data=processed_data)
# print (model)
exp = Experiment(args=args,model=model,data=processed_data)
exp.train()
exp.save_results()


print ("Time used :{:.01f}MINS".format((time.time()-start_time)/60))

print ("End time at : ")
print_time()

print ("************END***************")