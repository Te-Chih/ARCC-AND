import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, graph_layer, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.graph_layer = graph_layer

        self.graph_encoders = nn.ModuleList()
        if self.graph_layer == 1:
             self.graph_encoders.append(GCNConv(in_channels, out_channels))
        else:
            self.graph_encoders.append(GCNConv(in_channels, hidden))

            for _ in range(self.graph_layer - 2):
                self.graph_encoders.append(GCNConv(hidden, hidden))

            self.graph_encoders.append(GCNConv(hidden, out_channels))


    def forward(self, x, edge_index, edge_weight=None):
        if self.graph_layer == 1:
            x = self.graph_encoders[-1](x, edge_index, edge_weight)
            x = F.relu(x)
        else:
            # [0,n-2]
            for i, encoder in enumerate(self.graph_encoders[:-1]):
                x = encoder(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
            #[n-1] 层
            x = self.graph_encoders[-1](x, edge_index, edge_weight)
            x = F.relu(x)

        return x

class LowHigh_SR_CosineGraphLearnModule(nn.Module):
    def __init__(self,low_sim_threshold=0.20, high_sim_threshold=0.80):
        super(LowHigh_SR_CosineGraphLearnModule, self).__init__()
        self.low_threshold = low_sim_threshold
        self.high_threshold = high_sim_threshold
    def forward(self, semantic_x,model_x, adj):

        semantic_cos_sim = F.cosine_similarity(semantic_x.unsqueeze(1), semantic_x.unsqueeze(0), dim=2)
        model_cos_sim = F.cosine_similarity(model_x.unsqueeze(1), model_x.unsqueeze(0), dim=2)


        semantic_add_adj = torch.where(semantic_cos_sim>self.high_threshold, torch.ones_like(adj,requires_grad=semantic_cos_sim.requires_grad),torch.zeros_like(adj,requires_grad=semantic_cos_sim.requires_grad))
        model_add_adj = torch.where(model_cos_sim>self.high_threshold, torch.ones_like(adj,requires_grad=semantic_cos_sim.requires_grad),torch.zeros_like(adj,requires_grad=semantic_cos_sim.requires_grad))
        ms_add_adj = semantic_add_adj.int() & model_add_adj.int()


        semantic_rm_adj = torch.where(semantic_cos_sim<self.low_threshold, torch.ones_like(adj,requires_grad=semantic_cos_sim.requires_grad),torch.zeros_like(adj,requires_grad=semantic_cos_sim.requires_grad))
        model_rm_adj = torch.where(model_cos_sim<self.low_threshold, torch.ones_like(adj,requires_grad=semantic_cos_sim.requires_grad),torch.zeros_like(adj,requires_grad=semantic_cos_sim.requires_grad))
        ms_rm_adj = semantic_rm_adj.int() & model_rm_adj.int()

        # add edge
        new_adj_tmp = ms_add_adj | adj.int()

        # drop edge
        new_adj = new_adj_tmp ^ (new_adj_tmp & ms_rm_adj)

        refine_adj = new_adj - torch.diag_embed(torch.diag(new_adj))

        return refine_adj.float()



class LowHigh_S_CosineGraphLearnModule(nn.Module):
    def __init__(self,low_sim_threshold=0.20, high_sim_threshold=0.80):
        super(LowHigh_S_CosineGraphLearnModule, self).__init__()

        self.low_threshold = low_sim_threshold
        self.high_threshold = high_sim_threshold
    def drop_forward(self, semantic_x, adj):

        semantic_cos_sim = F.cosine_similarity(semantic_x.unsqueeze(1), semantic_x.unsqueeze(0), dim=2)


        semantic_add_adj = torch.where(semantic_cos_sim>self.high_threshold, torch.ones_like(adj,requires_grad=semantic_cos_sim.requires_grad),torch.zeros_like(adj,requires_grad=semantic_cos_sim.requires_grad))
        ms_add_adj = semantic_add_adj.int()

        semantic_rm_adj = torch.where(semantic_cos_sim<self.low_threshold, torch.ones_like(adj,requires_grad=semantic_cos_sim.requires_grad),torch.zeros_like(adj,requires_grad=semantic_cos_sim.requires_grad))
        ms_rm_adj = semantic_rm_adj.int()

        new_adj_tmp = ms_add_adj | adj.int()

        new_adj = new_adj_tmp ^ (new_adj_tmp & ms_rm_adj)



        refine_adj = new_adj - torch.diag_embed(torch.diag(new_adj))  # 对角线置 0



        return refine_adj.float()

    def forward(self, semantic_x, adj):
        semantic_cos_sim = F.cosine_similarity(semantic_x.unsqueeze(1), semantic_x.unsqueeze(0), dim=2)
        semantic_add_adj = torch.where(semantic_cos_sim > self.high_threshold,
                                       torch.ones_like(adj, requires_grad=semantic_cos_sim.requires_grad),
                                       adj)
        semantic_add_rm_adj = torch.where(semantic_cos_sim < self.low_threshold,
                                      torch.zeros_like(adj, requires_grad=semantic_cos_sim.requires_grad),
                                      semantic_add_adj)
        refine_adj = semantic_add_rm_adj - torch.diag_embed(torch.diag(semantic_add_rm_adj))  # 对角线置 0


        return refine_adj

class LowHigh_WS_CosineGraphLearnModule(nn.Module):
    def __init__(self,low_sim_threshold=0.20, high_sim_threshold=0.80):
        super(LowHigh_WS_CosineGraphLearnModule, self).__init__()

        self.low_threshold = low_sim_threshold
        self.high_threshold = high_sim_threshold
        self.weight1 = torch.nn.Linear(100, 100)
        self.weight2 = torch.nn.Linear(100, 100)
    def forward(self, semantic_x, adj):
        semantic_x = torch.tanh(self.weight1(semantic_x))
        semantic_x = torch.tanh(self.weight2(semantic_x))

        semantic_cos_sim = F.cosine_similarity(semantic_x.unsqueeze(1), semantic_x.unsqueeze(0), dim=2)

        semantic_add_adj = torch.where(semantic_cos_sim > self.high_threshold,
                                       torch.ones_like(adj, requires_grad=semantic_cos_sim.requires_grad),
                                       adj)
        semantic_add_rm_adj = torch.where(semantic_cos_sim < self.low_threshold,
                                          torch.zeros_like(adj, requires_grad=semantic_cos_sim.requires_grad),
                                          semantic_add_adj)
        refine_adj = semantic_add_rm_adj - torch.diag_embed(torch.diag(semantic_add_rm_adj))


        return refine_adj


class old_LowHigh_S_CosineGraphLearnModule(nn.Module):
    def __init__(self,low_sim_threshold=0.20, high_sim_threshold=0.80):
        super(old_LowHigh_S_CosineGraphLearnModule, self).__init__()

        self.low_threshold = low_sim_threshold
        self.high_threshold = high_sim_threshold
    def forward(self, semantic_x, adj):
        semantic_cos_sim = F.cosine_similarity(semantic_x.unsqueeze(1), semantic_x.unsqueeze(0), dim=2)

        semantic_add_adj = torch.where(semantic_cos_sim > self.high_threshold, torch.ones_like(adj),
                                       torch.zeros_like(adj))
        ms_add_adj = semantic_add_adj.int()

        semantic_rm_adj = torch.where(semantic_cos_sim < self.low_threshold, torch.ones_like(adj),
                                      torch.zeros_like(adj))
        ms_rm_adj = semantic_rm_adj.int()

        new_adj_tmp = ms_add_adj | adj.int()

        new_adj = new_adj_tmp ^ (new_adj_tmp & ms_rm_adj)

        refine_adj = new_adj - torch.diag_embed(torch.diag(new_adj))


        return refine_adj.float()

class old_LowHigh_WS_CosineGraphLearnModule(nn.Module):
    def __init__(self,low_sim_threshold=0.20, high_sim_threshold=0.80):
        super(old_LowHigh_WS_CosineGraphLearnModule, self).__init__()

        self.low_threshold = low_sim_threshold
        self.high_threshold = high_sim_threshold
        self.weight1 = torch.nn.Linear(100, 100)
        self.weight2 = torch.nn.Linear(100, 100)
    def forward(self, semantic_x, adj):
        semantic_x = torch.tanh(self.weight1(semantic_x))
        semantic_x = torch.tanh(self.weight2(semantic_x))

        semantic_cos_sim = F.cosine_similarity(semantic_x.unsqueeze(1), semantic_x.unsqueeze(0), dim=2)

        semantic_add_adj = torch.where(semantic_cos_sim > self.high_threshold, torch.ones_like(adj),
                                       torch.zeros_like(adj))
        ms_add_adj = semantic_add_adj.int()

        semantic_rm_adj = torch.where(semantic_cos_sim < self.low_threshold, torch.ones_like(adj),
                                      torch.zeros_like(adj))
        ms_rm_adj = semantic_rm_adj.int()

        new_adj_tmp = ms_add_adj | adj.int()

        new_adj = new_adj_tmp ^ (new_adj_tmp & ms_rm_adj)

        refine_adj = new_adj - torch.diag_embed(torch.diag(new_adj))



        return refine_adj.float()


class nofine_GraphLearnModule(nn.Module):
    def __init__(self,low_sim_threshold=0.20, high_sim_threshold=0.80):
        super(nofine_GraphLearnModule, self).__init__()

        self.low_threshold = low_sim_threshold
        self.high_threshold = high_sim_threshold

    def forward(self, semantic_x, adj):

        refine_adj = adj
        return refine_adj