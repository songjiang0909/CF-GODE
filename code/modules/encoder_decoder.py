import torch 
import torch.nn as nn
import torch.nn.functional as F
from modules.gcn import GCN


class Encoder(nn.Module):

    def __init__(self,input_dim,hidden,output_dim):
        super(Encoder,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, output_dim, bias=True)



    def forward(self,x):
        
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out



# class GEncoder(nn.Module):

#     def __init__(self,input_dim,hidden_dim,adj):
#         super(GEncoder,self).__init__()
        
#         self.net = GCN2(input_dim, hidden_dim) 
#         self.adj = adj

#     def _setting_(self,flag):

#         self.current_adj = self.adj[flag]

#     def forward(self,x):
        
#         out = self.net(x,self.current_adj)
        
#         return out





class Decoder(nn.Module):

    def __init__(self,input_dim,hidden,output_dim):
        super(Decoder,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, output_dim, bias=True)
        

    def forward(self,z):
        
        out = self.fc1(z)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out



class Decoder_A(nn.Module):

    def __init__(self,input_dim,hidden,output_dim):
        super(Decoder_A,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self,z):
        
        out = self.fc1(z)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


