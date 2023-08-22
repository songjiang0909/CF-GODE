import torch.nn as nn
import torch.nn.functional as F
from modules.component.graph_conv import GraphConvolution


class GCN(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_in, n_out)
        # self.dropout = dropout

    def forward(self, x, adj):

        out = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        return out+x[:,:-1]



class GCN2(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.5):
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(n_in, n_out)
        # self.dropout = dropout

    def forward(self, x, adj):

        out = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        return out+x


