import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from scipy import sparse as sp
from layers.san_gt_layer import SAN_GT_Layer
from layers.san_gt_lspe_layer import SAN_GT_LSPE_Layer
from layers.mlp_readout_layer import MLPReadout
from .sign_inv_net import get_sign_inv_net


class SANNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        full_graph = net_params['full_graph']
        init_gamma = net_params['init_gamma']
        # learn gamma
        self.gamma = nn.Parameter(torch.FloatTensor([init_gamma]))
        GT_layers = net_params['L']
        GT_hidden_dim = net_params['hidden_dim']
        GT_out_dim = net_params['out_dim']
        GT_n_heads = net_params['n_heads']
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.pe_init = net_params['pe_init']
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        self.pos_enc_dim = net_params['pos_enc_dim']
        self.use_lspe = net_params['use_lspe']
        if self.pe_init in ['rand_walk', 'lap_pe', 'map', 'map_ablation', 'signnet', 'oap', 'oap_ablation']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, GT_hidden_dim)
        self.embedding_h = nn.Embedding(num_atom_type, GT_hidden_dim)
        self.embedding_e = nn.Embedding(num_bond_type, GT_hidden_dim)
        if self.pe_init == 'rand_walk' or self.use_lspe:
            # LSPE
            self.layers = nn.ModuleList(
                [SAN_GT_LSPE_Layer(self.gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph,
                                   dropout, self.layer_norm, self.batch_norm, self.residual) for _ in
                 range(GT_layers - 1)])
            self.layers.append(SAN_GT_LSPE_Layer(self.gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph,
                                                 dropout, self.layer_norm, self.batch_norm, self.residual))
        else:
            # NoPE or LapPE or MAP or OAP
            self.layers = nn.ModuleList([SAN_GT_Layer(self.gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph,
                                                      dropout, self.layer_norm, self.batch_norm, self.residual) for _ in
                                         range(GT_layers - 1)])
            self.layers.append(SAN_GT_Layer(self.gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph,
                                            dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(GT_out_dim, 1)  # 1 out dim since regression problem
        if self.pe_init == 'rand_walk' or self.use_lspe:
            self.p_out = nn.Linear(GT_out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(GT_out_dim + self.pos_enc_dim, GT_out_dim)
        if self.use_lapeig_loss:
            self.p_out = nn.Linear(GT_out_dim, self.pos_enc_dim)
        self.g = None  # For util; To be accessed in loss() function
        if self.pe_init in ['map', 'map_ablation', 'signnet', 'oap', 'oap_ablation']:
            sign_inv_net = get_sign_inv_net(net_params)
            self.sign_inv_net = sign_inv_net
            self.pe_proj = nn.Linear(2 * GT_hidden_dim, GT_hidden_dim)

    def forward(self, g, h, p, e, snorm_n):
        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)
        h = self.in_feat_dropout(h)
        if self.pe_init in ['rand_walk', 'lap_pe', 'map', 'map_ablation', 'signnet', 'oap', 'oap_ablation']:
            p = self.embedding_p(p)
        if self.pe_init in ['lap_pe']:
            h = h + p
            if not self.use_lapeig_loss and not self.use_lspe:
                p = None
        if self.pe_init in ['map', 'map_ablation', 'signnet', 'oap', 'oap_ablation']:
            h = torch.cat([h, p], dim=1)
            h = self.pe_proj(h)
            if not self.use_lapeig_loss and not self.use_lspe:
                p = None
        for conv in self.layers:
            h, p = conv(g, h, p, e, snorm_n)
        g.ndata['h'] = h
        if self.pe_init == 'rand_walk' or self.use_lspe:
            p = self.p_out(p)
            g.ndata['p'] = p
        if self.pe_init == 'rand_walk' or self.use_lapeig_loss or self.use_lspe:
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means
            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p'] ** 2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms + 1e-6)
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p
        if self.pe_init == 'rand_walk' or self.use_lspe:
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'], g.ndata['p']), dim=-1))
            g.ndata['h'] = hp
        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        self.g = g  # For util; To be accessed in loss() function
        return self.MLP_layer(hg), g

    def loss(self, scores, targets):
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = nn.L1Loss()(scores, targets)
        if self.use_lapeig_loss:
            raise NotImplementedError
        else:
            loss = loss_a
        return loss
