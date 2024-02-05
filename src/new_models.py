import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv
from src.new_layers import *

class NB_AREByS_N2VEmbGCN_SCL(nn.Module):
    def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
        super(NB_AREByS_N2VEmbGCN_SCL, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gcn_layer = gcn_layer

        self.Sfc1 = nn.Linear(768, 200)
        self.Sfc2 = nn.Linear(200, 100)

        self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
        self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)

        self.Rfc1 = nn.Linear(100, hidden)
        self.Rfc2 = nn.Linear(hidden, 100)

        self.W = nn.ModuleList()
        for layer in range(self.gcn_layer):
            self.W.append(nn.Linear(100, 100))

        self.weight1 = torch.nn.Linear(100, 1)
        self.weight2 = torch.nn.Linear(100, 1)

        self.SRfc1 = nn.Linear(100, hidden)
        self.SRfc2 = nn.Linear(hidden, 100)
        if metric_type == "cosine":
            self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)

        if metric_type == "weighted_cosine":
            self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)

        if metric_type == 'old_cosine':
            self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)

        if metric_type == 'old_weighted_cosine':
            self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
        if metric_type == 'nofine':
            self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)


    def forward(self,pid_index_tensor,adj_matrix_tensor):

        sem_X1 = self.sem_emb_layer(pid_index_tensor)
        init_node_embeding = self.rel_emb_layer(pid_index_tensor)

        sem_X1 = self.dropout(sem_X1)
        sem_X1 = torch.relu(self.Sfc1(sem_X1))
        sem_X1_final= self.Sfc2(sem_X1)

        #refine graph structure
        refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)

        # add_adj_tensor add_adj_tensor.tolist()

        denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
        for l in range(self.gcn_layer):
            init_node_embeding = self.dropout(init_node_embeding)
            Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
            AxW = self.W[l](Ax)  ## N x m
            AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
            AxW = AxW / denom
            init_node_embeding = torch.relu(AxW)



        # rel_X1 =
        rel_X1 = self.dropout(init_node_embeding)
        rel_X1 = F.relu(self.Rfc1(rel_X1))
        rel_X1_final = self.Rfc2(rel_X1)

        weight1 = torch.sigmoid(self.weight1(sem_X1_final))
        weight2 = torch.sigmoid(self.weight2(rel_X1_final))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        output = weight1 * sem_X1_final + weight2 * rel_X1_final

        output = self.dropout(output)
        output = torch.relu(self.SRfc1(output))
        output = self.SRfc2(output)


        return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor

# class NB_AREBySR_N2VEmbGCN_SCL(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(NB_AREBySR_N2VEmbGCN_SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#         self.weight1 = torch.nn.Linear(100, 1)
#         self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == "cosine_SR":
#             self.structureLearn = LowHigh_SR_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#
#         #todo refine graph structure
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,init_node_embeding,adj_matrix_tensor)
#
#
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         weight1 = torch.sigmoid(self.weight1(sem_X1_final))
#         weight2 = torch.sigmoid(self.weight2(rel_X1_final))
#         weight1 = weight1 / (weight1 + weight2)
#         weight2 = 1 - weight1
#
#         output = weight1 * sem_X1_final + weight2 * rel_X1_final
#
#         output = self.dropout(output)
#
#         output = torch.relu(self.SRfc1(output))
#         output = self.SRfc2(output)
#
#
#
#
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
# class NB_AREByS_N2VEmbGCN_SCL_GSL(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(NB_AREByS_N2VEmbGCN_SCL_GSL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#         self.weight1 = torch.nn.Linear(100, 1)
#         self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         weight1 = torch.sigmoid(self.weight1(sem_X1_final))
#         weight2 = torch.sigmoid(self.weight2(rel_X1_final))
#         weight1 = weight1 / (weight1 + weight2)
#         weight2 = 1 - weight1
#
#         output = weight1 * sem_X1_final + weight2 * rel_X1_final
#
#         output = self.dropout(output)
#
#         output = torch.relu(self.SRfc1(output))
#         output = self.SRfc2(output)
#
#
#
#
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
#
#
# class NB_2MLP_1SCL(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(NB_2MLP_1SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#
#         refine_adj_matrix_tensor=adj_matrix_tensor
#
#
#
#         return sem_X1_final,refine_adj_matrix_tensor
#
# class Rule_N2VGCN_2MLP_1SCL(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(Rule_N2VGCN_2MLP_1SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#
#         refine_adj_matrix_tensor=adj_matrix_tensor
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#
#
#
#         return rel_X1_final,refine_adj_matrix_tensor
#
#
# class NB_AREByS_N2VEmbGCN_concat_3SCL(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(NB_AREByS_N2VEmbGCN_concat_3SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#
#         self.SRfc1 = nn.Linear(200, 200)
#         self.SRfc2 = nn.Linear(200, 200)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         output = torch.cat([sem_X1_final, rel_X1_final], 1)
#
#
#         output = self.dropout(output)
#
#         output = torch.relu(self.SRfc1(output))
#         output = self.SRfc2(output)
#
#
#
#
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
#
# class BERTEmbGCN_SCL(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(BERTEmbGCN_SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         # self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         # self.Rfc1 = nn.Linear(100, hidden)
#         # self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#         # self.weight1 = torch.nn.Linear(100, 1)
#         # self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#         # elif metric_type == "dorpEdge":
#         #     self.structureLearn = DropEdge_CosineGraphLearnModule(sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         # init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#
#         #todo refine graph structure
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#         # add_adj_tensor = refine_adj_matrix_tensor - adj_matrix_tensor
#         # add_adj_tensor add_adj_tensor.tolist()
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             sem_X1_final = self.dropout(sem_X1_final)
#             Ax = refine_adj_matrix_tensor.mm(sem_X1_final)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](sem_X1_final)  ## self loop  N x h
#             AxW = AxW / denom
#             sem_X1_final = torch.relu(AxW)
#
#
#
#
#
#         output = self.dropout(sem_X1_final)
#
#         output = torch.relu(self.SRfc1(output))
#         output = self.SRfc2(output)
#
#
#
#
#         return output,refine_adj_matrix_tensor
#
#
# class NB_AREByS_N2VEmbGCN_2SCL_concat(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(NB_AREByS_N2VEmbGCN_2SCL_concat, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#         # elif metric_type == "dorpEdge":
#         #     self.structureLearn = DropEdge_CosineGraphLearnModule(sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#
#         #todo refine graph structure
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#         # add_adj_tensor = refine_adj_matrix_tensor - adj_matrix_tensor
#         # add_adj_tensor add_adj_tensor.tolist()
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         output = torch.cat([sem_X1_final, rel_X1_final], 1)
#
#
#
#
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
# class NB_AREByS_N2VEmbGCN_2SCL_ADD(nn.Module):
#     def __init__(self,sem_freeze,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(NB_AREByS_N2VEmbGCN_2SCL_concat, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#         # elif metric_type == "dorpEdge":
#         #     self.structureLearn = DropEdge_CosineGraphLearnModule(sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#         #todo refine graph structure
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#         # add_adj_tensor = refine_adj_matrix_tensor - adj_matrix_tensor
#         # add_adj_tensor add_adj_tensor.tolist()
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         output = torch.cat([sem_X1_final, rel_X1_final], 1)
#
#         # weight1 = torch.sigmoid(self.weight1(sem_X1_final))
#         # weight2 = torch.sigmoid(self.weight2(rel_X1_final))
#         # weight1 = weight1 / (weight1 + weight2)
#         # weight2 = 1 - weight1
#         #
#         # output = weight1 * sem_X1_final + weight2 * rel_X1_final
#         #
#         # output = self.dropout(output)
#         #
#         # output = torch.relu(self.SRfc1(output))
#         # output = self.SRfc2(output)
#
#
#
#
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
# class NB_AREByS_N2VEmbGCN_SCL_BCE(nn.Module):
#     def __init__(self, sem_freeze, rel_freeze, sem_emb_vector, rel_emb_vector, hidden=100, dropout=0.5, gcn_layer=1,
#                  low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(NB_AREByS_N2VEmbGCN_SCL_BCE, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector, freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector, freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#         self.weight1 = torch.nn.Linear(100, 1)
#         self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#
#         self.SRfc3 = nn.Linear(200, hidden)
#         self.SRfc4 = nn.Linear(hidden, 1)
#
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         # elif metric_type == "dorpEdge":
#         #     self.structureLearn = DropEdge_CosineGraphLearnModule(sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#     def forward(self, pid_index_tensor, adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final = self.Sfc2(sem_X1)
#
#
#         refine_adj_matrix_tensor = self.structureLearn(sem_X1_final, adj_matrix_tensor)
#
#         # add_adj_tensor = refine_adj_matrix_tensor - adj_matrix_tensor
#         # add_adj_tensor add_adj_tensor.tolist()
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         weight1 = torch.sigmoid(self.weight1(sem_X1_final))
#         weight2 = torch.sigmoid(self.weight2(rel_X1_final))
#         weight1 = weight1 / (weight1 + weight2)
#         weight2 = 1 - weight1
#
#         output = weight1 * sem_X1_final + weight2 * rel_X1_final
#
#         output = self.dropout(output)
#
#         output = torch.relu(self.SRfc1(output))
#         output_scl = self.SRfc2(output)
#         # 构造paperpair
#         # 构造不同的论文对,一个name下共p篇论文,则可以构成p*p篇论文对，即[p,p]
#         # [p,d] -> [1, p, d] -> [p, p, d] ：将[p,d] 复制p份； 生成p份该name下的全部论文的向量,(用这个作为，paper2)
#         h_0s = output_scl.unsqueeze(dim=0).expand(output_scl.shape[0], output_scl.shape[0], 100)
#         # [p,d] -> [p, 1, d] -> [p, p, d] ： 在每p里，将d复制p份； 生成p份某一个论文向量；（用这个作为，paper1）；
#         # 用某一个论文向量和全部的论文向量组成了p对 <paper1【p份某论文向量】,paper2【该name下全部论文向量共p个】>
#         h_1s = output_scl.unsqueeze(dim=1).expand(output_scl.shape[0], output_scl.shape[0], 100)
#
#         #[p,p,2d]
#         output = torch.cat([h_1s, h_0s], 2)
#         new_output = output.view(-1, output.shape[2])
#
#         new_output = F.relu(self.SRfc3(new_output))
#         output_bce = self.SRfc4(new_output).squeeze()
#
#         return sem_X1_final, rel_X1_final, output_scl, output_bce,refine_adj_matrix_tensor
#
#
# class WIW_NB_AREByS_N2VEmbGCN_SCL(nn.Module):
#     def __init__(self,sem_freeze,semantic_type,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(WIW_NB_AREByS_N2VEmbGCN_SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#         if semantic_type == "W2V":
#             self.Sfc1 = nn.Linear(100, hidden)
#             self.Sfc2 = nn.Linear(hidden, 100)
#         else:
#             self.Sfc1 = nn.Linear(768, 200)
#             self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#         self.weight1 = torch.nn.Linear(100, 1)
#         self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#         # elif metric_type == "dorpEdge":
#         #     self.structureLearn = DropEdge_CosineGraphLearnModule(sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#         # add_adj_tensor = refine_adj_matrix_tensor - adj_matrix_tensor
#         # add_adj_tensor add_adj_tensor.tolist()
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         weight1 = torch.sigmoid(self.weight1(sem_X1_final))
#         weight2 = torch.sigmoid(self.weight2(rel_X1_final))
#         weight1 = weight1 / (weight1 + weight2)
#         weight2 = 1 - weight1
#
#         output = weight1 * sem_X1_final + weight2 * rel_X1_final
#
#         output = self.dropout(output)
#
#         output = torch.relu(self.SRfc1(output))
#         output = self.SRfc2(output)
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
#
#
# class WIW_NB_1SCL(nn.Module):
#     def __init__(self,sem_freeze,semantic_type,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(WIW_NB_1SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#         if semantic_type == "W2V":
#             self.Sfc1 = nn.Linear(100, hidden)
#             self.Sfc2 = nn.Linear(hidden, 100)
#         else:
#             self.Sfc1 = nn.Linear(768, 200)
#             self.Sfc2 = nn.Linear(200, 100)
#
#         # self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#         self.weight1 = torch.nn.Linear(100, 1)
#         self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#         # elif metric_type == "dorpEdge":
#         #     self.structureLearn = DropEdge_CosineGraphLearnModule(sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         # init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#         return sem_X1_final, adj_matrix_tensor
#
#
# class WIW_Rule_N2VGCN2MLP_1SCL(nn.Module):
#     def __init__(self,sem_freeze,semantic_type,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(WIW_Rule_N2VGCN2MLP_1SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#         if semantic_type == "W2V":
#             self.Sfc1 = nn.Linear(100, hidden)
#             self.Sfc2 = nn.Linear(hidden, 100)
#         else:
#             self.Sfc1 = nn.Linear(768, 200)
#             self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         # self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         # sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#
#         # todo refine graph structure
#         refine_adj_matrix_tensor=adj_matrix_tensor
#
#         # add_adj_tensor = refine_adj_matrix_tensor - adj_matrix_tensor
#         # add_adj_tensor add_adj_tensor.tolist()
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         return rel_X1_final, adj_matrix_tensor
#
#
# class WIW_NB_AREByS_N2VEmbGCN_Concat_SCL(nn.Module):
#     def __init__(self,sem_freeze,semantic_type,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(WIW_NB_AREByS_N2VEmbGCN_Concat_SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#         if semantic_type == "W2V":
#             self.Sfc1 = nn.Linear(100, hidden)
#             self.Sfc2 = nn.Linear(hidden, 100)
#         else:
#             self.Sfc1 = nn.Linear(768, 200)
#             self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#
#
#         self.SRfc1 = nn.Linear(200, 200)
#         self.SRfc2 = nn.Linear(200, 200)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#         # elif metric_type == "dorpEdge":
#         #     self.structureLearn = DropEdge_CosineGraphLearnModule(sim_threshold)
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#         # add_adj_tensor = refine_adj_matrix_tensor - adj_matrix_tensor
#         # add_adj_tensor add_adj_tensor.tolist()
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#
#         output = torch.cat([sem_X1_final,rel_X1_final],1)
#
#         output = self.dropout(output)
#
#         output = torch.relu(self.SRfc1(output))
#         output = self.SRfc2(output)
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
#
# class WIW_NB_AREByS_N2VEmbGCN_SCL_CAT(nn.Module):
#     def __init__(self,sem_freeze,semantic_type,rel_freeze,sem_emb_vector,rel_emb_vector,hidden=100, dropout=0.5,gcn_layer= 1,low_sim_threshold=0.50, high_sim_threshold=0.9, metric_type="cosine"):
#         super(WIW_NB_AREByS_N2VEmbGCN_SCL_CAT, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#         if semantic_type == "W2V":
#             self.Sfc1 = nn.Linear(100, hidden)
#             self.Sfc2 = nn.Linear(hidden, 100)
#         else:
#             self.Sfc1 = nn.Linear(768, 200)
#             self.Sfc2 = nn.Linear(200, 100)
#
#         self.rel_emb_layer = nn.Embedding.from_pretrained(rel_emb_vector,freeze=rel_freeze)
#         self.sem_emb_layer = nn.Embedding.from_pretrained(sem_emb_vector,freeze=sem_freeze)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#         self.W = nn.ModuleList()
#         for layer in range(self.gcn_layer):
#             self.W.append(nn.Linear(100, 100))
#
#         self.weight1 = torch.nn.Linear(100, 1)
#         self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#         if metric_type == "cosine":
#             self.structureLearn = LowHigh_S_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#             # self.structureLearn = High_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#         if metric_type == "weighted_cosine":
#             self.structureLearn = LowHigh_WS_CosineGraphLearnModule(low_sim_threshold,high_sim_threshold)
#
#
#         if metric_type == 'old_cosine':
#             self.structureLearn = old_LowHigh_S_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#         if metric_type == 'old_weighted_cosine':
#             self.structureLearn = old_LowHigh_WS_CosineGraphLearnModule(low_sim_threshold, high_sim_threshold)
#         if metric_type == 'nofine':
#             self.structureLearn = nofine_GraphLearnModule(low_sim_threshold, high_sim_threshold)
#
#
#     def forward(self,pid_index_tensor,adj_matrix_tensor):
#
#         sem_X1 = self.sem_emb_layer(pid_index_tensor)
#         init_node_embeding = self.rel_emb_layer(pid_index_tensor)
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#
#
#         refine_adj_matrix_tensor=self.structureLearn(sem_X1_final,adj_matrix_tensor)
#
#
#
#         denom = refine_adj_matrix_tensor.sum(1).unsqueeze(1) + 1
#         for l in range(self.gcn_layer):
#             init_node_embeding = self.dropout(init_node_embeding)
#             Ax = refine_adj_matrix_tensor.mm(init_node_embeding)  ## N x N  times N x h  = Nxh
#             AxW = self.W[l](Ax)  ## N x m
#             AxW = AxW + self.W[l](init_node_embeding)  ## self loop  N x h
#             AxW = AxW / denom
#             init_node_embeding = torch.relu(AxW)
#
#
#
#         # rel_X1 =
#         rel_X1 = self.dropout(init_node_embeding)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#         output = torch.cat([sem_X1_final, rel_X1_final], 1)
#
#         return sem_X1_final, rel_X1_final ,output,refine_adj_matrix_tensor
#
#
# class NB_R_GCN_SCL(nn.Module):
#     def __init__(self, hidden=100, dropout=0.5,gcn_layer= 2):
#         super(NB_R_GCN_SCL, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.gcn_layer = gcn_layer
#
#         self.Sfc1 = nn.Linear(768, 200)
#         self.Sfc2 = nn.Linear(200, 100)
#
#         self.gcn =  GCN(100, hidden, 100, self.gcn_layer, dropout)
#
#         self.Rfc1 = nn.Linear(100, hidden)
#         self.Rfc2 = nn.Linear(hidden, 100)
#
#
#
#         self.weight1 = torch.nn.Linear(100, 1)
#         self.weight2 = torch.nn.Linear(100, 1)
#
#         self.SRfc1 = nn.Linear(100, hidden)
#         self.SRfc2 = nn.Linear(hidden, 100)
#
#     def forward(self, batch):
#         sem_X1 = batch.s
#         init_node_embedding = batch.x
#         edge_index = batch.edge_index
#         edge_weight = None
#
#
#         sem_X1 = self.dropout(sem_X1)
#         sem_X1 = torch.relu(self.Sfc1(sem_X1))
#         sem_X1_final= self.Sfc2(sem_X1)
#
#         rel_X1 = self.gcn(init_node_embedding,edge_index)
#
#
#
#         rel_X1 = self.dropout(rel_X1)
#         rel_X1 = F.relu(self.Rfc1(rel_X1))
#         rel_X1_final = self.Rfc2(rel_X1)
#
#         weight1 = torch.sigmoid(self.weight1(sem_X1_final))
#         weight2 = torch.sigmoid(self.weight2(rel_X1_final))
#         weight1 = weight1 / (weight1 + weight2)
#         weight2 = 1 - weight1
#
#         output = weight1 * sem_X1_final + weight2 * rel_X1_final
#
#         output = self.dropout(output)
#         output = torch.relu(self.SRfc1(output))
#         output = self.SRfc2(output)
#         return sem_X1_final, rel_X1_final ,output

