
import torch
import torch.nn as nn
from modules.gcn import GCN



class GraphODEFunc(nn.Module):
    
    def __init__(self, args,hidden_dim,adj,treatments,cf_treatments):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(GraphODEFunc, self).__init__()

        self.args = args
        self.adj = adj
        self.treatments = treatments
        self.cf_treatments = cf_treatments
        self.ode_func_net = GCN(hidden_dim+1, hidden_dim)  
        self.nfe = 0


    def _setting_(self,flag,fcf):

        self.current_adj = self.adj[flag]
        if fcf == "f":
            self.current_treatments = self.treatments[flag]
        if fcf == "cf":
            self.current_treatments = self.cf_treatments[flag]


    def forward(self, t, z):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """

        a_index = int(t*(self.args.num_time_steps-1))
        a_t = self.current_treatments[:,a_index,:]
        # print ("a_t:{}".format(a_t.shape))
        # print ("a_index:{}".format(a_index))
        # print ("current_treatments:{}".format(self.current_treatments.shape))
        z_cat = torch.cat((z,a_t),1)
        self.nfe += 1
        #print(self.nfe)
        grad = self.ode_func_net(z_cat,self.current_adj)


        return grad

    

