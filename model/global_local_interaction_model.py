#coding=utf-8
import sys
import os
import torch
import copy
from torch import nn
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool, MLP
from .baseline_models import Baseline_Models, embed_atom_chem, embed_bond_chem, get_interaction

class GlobalLocalInteractionModel(nn.Module):
    def __init__(self,
            chem_info_embedding_model_type="gat_gcn",
            global_interaction=0,
            local_interaction=0,
            local_interaction_cutoff=5,
            local_interaction_model_type=None,
            out_depth=3,
            out_features=128):
        super(GlobalLocalInteractionModel, self).__init__()
        self.chem_info_embedding_model_type = chem_info_embedding_model_type
        self.global_interaction = global_interaction
        self.local_interaction = local_interaction
        self.local_interaction_cutoff = local_interaction_cutoff
        self.local_interaction_model_type = local_interaction_model_type
        atom_channels = 16
        bond_channels = 16
        # set chem_info_embedding_model
        # gat_gcn
        if self.chem_info_embedding_model_type == "gat_gcn":
            self.chem_info_embedding_model = Baseline_Models(atom_channels=atom_channels, bond_channels=bond_channels, out_features=out_features, gat_depth=1, gcn_depth=3)
        # gat
        if self.chem_info_embedding_model_type == "gat":
            self.chem_info_embedding_model = Baseline_Models(atom_channels=atom_channels, bond_channels=bond_channels, out_features=out_features, gat_depth=1, gcn_depth=0)
        # gcn
        if self.chem_info_embedding_model_type == "gcn":
            self.chem_info_embedding_model = Baseline_Models(atom_channels=atom_channels, bond_channels=bond_channels, out_features=out_features, gat_depth=0, gcn_depth=3)
        # gat_gin
        if self.chem_info_embedding_model_type == "gat_gin":
            self.chem_info_embedding_model = Baseline_Models(atom_channels=atom_channels, bond_channels=bond_channels, out_features=out_features, gat_depth=1, gcn_depth=0, gin_depth=3)
        # gin
        if self.chem_info_embedding_model_type == "gin":
            self.chem_info_embedding_model = Baseline_Models(atom_channels=atom_channels, bond_channels=bond_channels, out_features=out_features, gat_depth=0, gcn_depth=0, gin_depth=3)
        # gat_gcn2
        if self.chem_info_embedding_model_type == "gat_gcn2":
            self.chem_info_embedding_model = Baseline_Models(atom_channels=atom_channels, bond_channels=bond_channels, out_features=out_features, gat_depth=1, gcn2_depth=3)
        # gcn2
        if self.chem_info_embedding_model_type == "gcn2":
            self.chem_info_embedding_model = Baseline_Models(atom_channels=atom_channels, bond_channels=bond_channels, out_features=out_features, gat_depth=0, gcn2_depth=3)
        # chem info out
        if (self.global_interaction == 0) and (self.local_interaction == 0):
            self.chem_info_node_out = nn.Sequential()
            for i in range(out_depth):
                self.chem_info_node_out.add_module("chem_info_node_out_%s" % i, nn.Linear(in_features=out_features, out_features=out_features))
                self.chem_info_node_out.add_module("chem_info_node_out_relu_%s" % i, nn.ReLU())
            self.chem_info_out = Linear(in_features=out_features, out_features=1)

        # global interaction
        if self.global_interaction == 1:
            # ligand_out
            self.ligand_out = nn.Sequential()
            self.ligand_out.add_module("ligand_out", MLP(in_channels=out_features,
                hidden_channels=out_features, out_channels=out_features, dropout=0.1, num_layers=out_depth))
            # protein_out
            self.protein_out = nn.Sequential()
            self.protein_out.add_module("protein_out", MLP(in_channels=out_features,
                hidden_channels=out_features, out_channels=out_features, dropout=0.1, num_layers=out_depth))
            # protein_ligand_interaction
            self.protein_ligand_out = nn.Sequential()
            for i in range(out_depth):
                self.protein_ligand_out.add_module("protein_ligand_out_%s" % i,
                    nn.Linear(in_features=out_features * 2, out_features=out_features * 2))
                self.protein_ligand_out.add_module("protein_ligand_relu_%s" % i, nn.ReLU())
            self.protein_ligand_out.add_module("protein_ligand_out",
                nn.Linear(in_features=out_features*2, out_features=1))

            # relative position attention
            self.cal_attention = nn.Sequential()
            attention_in_features = 42
            attention_in_features += out_features
            mlp = MLP(in_channels=attention_in_features,
                    hidden_channels=out_features,
                    out_channels=1,
                    dropout=0.1,
                    num_layers=out_depth)
            self.cal_attention.add_module("cal_attention",  mlp)
            self.cal_attention.add_module("sigmoid", nn.Sigmoid())

        # local interaction
        if self.local_interaction == 1:
            if self.local_interaction_model_type == None:
                self.local_interaction_model = lambda batch_data: batch_data.x
            elif self.local_interaction_model_type == "SchNet":
                self.local_interaction_model = SchNet(hidden_channels=out_features, cutoff=self.local_interaction_cutoff)
            elif self.local_interaction_model_type == "DimeNet":
                self.local_interaction_model = DimeNet(hidden_channels=out_features, out_channels=out_features,
                        cutoff=self.local_interaction_cutoff, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6)

            self.local_interaction_out = nn.Sequential()
            for i in range(out_depth):
                self.local_interaction_out.add_module("interaction_out_%s" % i,
                    nn.Linear(in_features=out_features * 2 + 1, out_features=out_features * 2 + 1))
                self.local_interaction_out.add_module("interaction_relu_%s" % i, nn.ReLU())
            self.local_interaction_out.add_module("interaction_out",
                nn.Linear(in_features=out_features * 2 + 1, out_features=1))

    def forward(self, batch_data):
        result = 0
        chem_info_result = 0
        global_interaction_result = 0
        local_interaction_result = 0

        # embedding data
        batch = batch_data.batch
        node_embedding = self.chem_info_embedding_model(batch_data)

        # chem_info_result
        if (self.global_interaction == 0) and (self.local_interaction == 0):
            node_embedding_global = self.chem_info_node_out(node_embedding)
            node_embedding_global = global_add_pool(node_embedding_global, batch)
            chem_info_result = self.chem_info_out(node_embedding_global)

        # global_interaction_result
        if self.global_interaction == 1:
            # split ligand nodes and protein nodes
            # ligand nodes: batch_data.node_type == 0
            # protein nodes: batch_data.node_type == 1
            # ligand
            node_embedding_ligand = self.ligand_out(node_embedding[batch_data.node_type == 0])
            # aggregate ligand nodes embedding
            node_embedding_ligand = global_add_pool(node_embedding_ligand,
                    batch=batch[batch_data.node_type == 0])
            # protein
            node_embedding_protein = self.protein_out(node_embedding[batch_data.node_type == 1])
            # relative position attention
            cutoff_data = torch.cat([node_embedding_protein, batch_data.all_cutoff_data[batch_data.node_type == 1]], axis=1)
            protein_attention = self.cal_attention(cutoff_data)
            node_embedding_protein = node_embedding_protein * protein_attention
            # aggregate protein nodes embedding
            node_embedding_protein = global_add_pool(node_embedding_protein,
                    batch=batch[batch_data.node_type == 1])
            global_interaction_embedding = torch.cat([node_embedding_ligand, node_embedding_protein], axis=1)
            # get global interaction result
            global_interaction_result = self.protein_ligand_out(global_interaction_embedding)

        # local_interaction_result
        if self.local_interaction == 1:
            # get local_interaction_graph
            new_result, new_batch, new_batch_data = get_interaction(batch_data,
                    node_embedding,
                    cutoff=self.local_interaction_cutoff,
                    return_new_batch_data=1)

            if self.local_interaction_model_type == None:
                local_node_embedding = self.local_interaction_model(new_batch_data)
            else:
                local_node_embedding = self.local_interaction_model(z=new_batch_data.x, pos=new_batch_data.pos, batch=new_batch_data.batch)
            # update node embedding in local graph
            local_new_node_embedding, local_new_batch, _ = get_interaction(new_batch_data,
                    local_node_embedding,
                    cutoff=self.local_interaction_cutoff,
                    return_new_batch_data=0)
            local_new_result = self.local_interaction_out(local_new_node_embedding)
            # get local interaction result
            local_interaction_result = global_add_pool(local_new_result, local_new_batch)

        # aggregate result
        if (self.global_interaction == 0) and (self.local_interaction == 0):
            result = chem_info_result
        else:
            result = global_interaction_result + local_interaction_result
        return result
