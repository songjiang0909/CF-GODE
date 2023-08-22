import random
import os
import pickle as pkl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint as odeint
from modules.encoder_decoder import Encoder, Decoder,Decoder_A
from modules.graphODE import GraphODEFunc
# from modules.component.grl import GradReverse


class CGODE(nn.Module):
    
    def __init__(self,args,data):
        super(CGODE, self).__init__()

        self.FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

        self.args = args
        self.data = data

        self.train_adj = data["train_outputs"]["adj"]
        self.val_adj = data["val_outputs"]["adj"]
        self.test_adj = data["test_outputs"]["adj"]
        self.adj = {"train":self.train_adj,"val":self.val_adj,"test":self.test_adj}

        self.train_treatments = data["train_outputs"]["action_application_point"]
        self.val_treatments = data["val_outputs"]["action_application_point"]
        self.test_treatments = data["test_outputs"]["action_application_point"]
        self.treatments = {"train":self.train_treatments,"val":self.val_treatments,"test":self.test_treatments}


        self.cf_train_treatments = data["train_outputs"]["cf_action_application_point"]
        self.cf_val_treatments = data["val_outputs"]["cf_action_application_point"]
        self.cf_test_treatments = data["test_outputs"]["cf_action_application_point"]
        self.cf_treatments = {"train":self.cf_train_treatments,"val":self.cf_val_treatments,"test":self.cf_test_treatments}

        self.train_feature = data["train_outputs"]["feature"]
        self.val_feature = data["val_outputs"]["feature"]
        self.test_feature = data["test_outputs"]["feature"]
        self.feature = {"train":self.train_feature,"val":self.val_feature,"test":self.test_feature}




        self.odeint_rtol = args.odeint_rtol
        self.odeint_atol = args.odeint_atol
        self.ode_method = args.ode_method

        # self.input_dim = data["train_outputs"]["health_condition"].shape[2]
        # self.input_dim = data["train_outputs"]["feature"].shape[2]
        self.input_dim = data["train_outputs"]["health_condition"].shape[2] + data["train_outputs"]["feature"].shape[2]
        self.hidden_dim = self.args.hidden_dim
        self.output_dim = 1
        self.time_steps = args.num_time_steps
        self.interval = self.FloatTensor([i/(self.time_steps-1) for i in range(self.time_steps)])
        

        self.encoder = Encoder(self.input_dim,self.hidden_dim,self.hidden_dim)
        self.ode_func = GraphODEFunc(self.args,self.hidden_dim,self.adj,self.treatments,self.cf_treatments)
        self.transform = Decoder(self.hidden_dim+10,self.hidden_dim,self.hidden_dim)
        
        self.outcome = Decoder(self.hidden_dim,self.hidden_dim,1)
        self.treatment_net = Decoder_A(self.hidden_dim,self.hidden_dim,1)
        self.neighbor_net = Decoder(self.hidden_dim+1,self.hidden_dim,1)
        
        # self.transform_nei = Decoder(self.hidden_dim+1,self.hidden_dim,self.hidden_dim)
        # self.neighbor_net = Decoder(self.hidden_dim,self.hidden_dim,1)
        # self.grl_a = GradientReversal()
        # self.grl_s = GradientReversal()
    

    def __set_feature__(self,flag):

        return self.feature[flag]
    
    def forward(self,x,a,flag,fcf):

        # print ("==="*20)
        # print ("x_shape:{}".format(x.shape))
        z0 = self.encoder(x) 
        # print ("z0_shape:{}".format(z0.shape))


        self.ode_func._setting_(flag,fcf)

        z_hat = odeint(self.ode_func, z0, self.interval, rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method) 

        # print ("z_hat_shape:{}".format(z_hat.shape)) 
        z_hat = z_hat.permute(1,0,2) #Nsample * L * hidden dim

        # feature = self.__set_feature__(flag)
        # z_hat = self.transform(torch.cat((z_hat,feature[:,:,:]),2))

        # print ("z_hat_shape after permute:{}".format(z_hat.shape))
        pred_y = self.outcome(z_hat) 
        # print ("pred_y_shape:{}".format(pred_y.shape))

        z_hat_ra = gradient_reversal_layer(z_hat)
        pred_a = self.treatment_net(z_hat_ra)

        # print ("z_hat_ra_shape:{}".format(z_hat_ra.shape))
        # print ("pred_a_shape:{}".format(pred_a.shape))

        z_hat_rs = gradient_reversal_layer(z_hat)
        z_hat_rs = torch.cat((z_hat_rs,a),2)
        pred_s = self.neighbor_net(z_hat_rs)

        # print ("z_hat_rs_shape:{}".format(z_hat_rs.shape))
        # print ("pred_s_shape:{}".format(pred_s.shape))

        return pred_y,pred_a,pred_s,z_hat,z_hat_rs





class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return - grad_output

def gradient_reversal_layer(x):
    return GradientReversalLayer.apply(x)