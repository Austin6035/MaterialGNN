# 模型文件
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn

class GCN_2(torch.nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_nbr):
        super(GCN_2, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.num_nbr = num_nbr
        self.bn1 = nn.BatchNorm1d(self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.bn3 = nn.BatchNorm1d(self.nbr_fea_len)
        self.bn4 = nn.BatchNorm1d(self.atom_fea_len)

        self.fc_core = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, self.atom_fea_len)
        self.fc_filter = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, self.atom_fea_len)
        self.fc_bond = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, self.nbr_fea_len)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, ):
        atom_nbr_fea = x[edge_index[1], :]
        atom_init_fea = x[edge_index[0], :]
        Z = torch.cat((atom_nbr_fea, atom_init_fea, edge_attr), dim=1)

        a_filter = self.bn1(self.fc_core(Z))
        a_core = self.bn2(self.fc_filter(Z))
        bond = self.bn3(self.fc_bond(Z))

        a_filter = torch.sigmoid(a_filter)
        a_core = F.softplus(a_core)

        nbr_sumed = a_filter * a_core
        nbr_sumed = nbr_sumed.reshape((-1, self.num_nbr, self.atom_fea_len))
        nbr_sumed = torch.sum(nbr_sumed, dim=1)
        nbr_sumed = self.bn4(nbr_sumed)
        out = F.softplus(x + nbr_sumed)
        bond_out = F.softplus(edge_attr + bond)
        return out, bond_out


class GCN_Layer(torch.nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_nbr):
        super(GCN_Layer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.num_nbr = num_nbr
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, ):
        atom_nbr_fea = x[edge_index[1], :]
        atom_init_fea = x[edge_index[0], :]
        total_nbr_fea = torch.cat((atom_nbr_fea, atom_init_fea, edge_attr), dim=1)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=1)
        nbr_filter = torch.sigmoid(nbr_filter)
        nbr_core = F.softplus(nbr_core)

        nbr_sumed = nbr_core * nbr_filter
        nbr_sumed = nbr_sumed.reshape((-1, self.num_nbr, self.atom_fea_len))
        nbr_sumed = torch.sum(nbr_sumed, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = F.softplus(x + nbr_sumed)
        return out


class AtomEmbedding(torch.nn.Module):

    def __init__(self, emb_dim, properties_list=None):
        super(AtomEmbedding, self).__init__()
        self.properties_list = properties_list
        self.properties_name = ['N', 'G', 'P', 'NV', 'E', 'R', 'V', 'EA', 'I']
        self.full_dims = [100, 18, 7, 12, 10, 10, 10, 10, 10]
        full_atom_feature_dims = self.__get_full_dims()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            # 调节方差
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding

    def __get_full_dims(self):
        feature_dim = []
        if self.properties_list == 'all':
            feature_dim = self.full_dims
        elif len(self.properties_list) == 1 or self.properties_list[0] == 'N':
            feature_dim = [100]
        else:
            for prop in self.properties_list:
                index = self.properties_name.index(prop)
                feature_dim.append(self.full_dims[index])
        return feature_dim


class Net(torch.nn.Module):

    def __init__(self, orig_bond_fea_len=51, nbr_fea_len=128, atom_fea_len=64, n_conv=3, h_fea_len=128, l1=1,
                 l2=1, classification=False, n_classes=2, gcn1=False, attention=False, dynamic_attention=False, n_heads=1,
                 max_num_nbr=12, pooling='mean', p=0, properties_list=None):
        super(Net, self).__init__()
        self.classification = classification
        self.pooling = pooling
        self.bn = nn.BatchNorm1d(atom_fea_len)
        self.atom_embedding = AtomEmbedding(atom_fea_len, properties_list=properties_list)
        self.bond_embedding = nn.Embedding(orig_bond_fea_len, nbr_fea_len)

        self.p = p
        if attention:
            self.n_convs = nn.ModuleList([torch_geometric.nn.GATConv(atom_fea_len, atom_fea_len,
                                                                     edge_dim=nbr_fea_len, heads=n_heads, concat=False)
                                          for _ in range(n_conv)])
        elif dynamic_attention:
            self.n_convs = nn.ModuleList([torch_geometric.nn.GATv2Conv(atom_fea_len, atom_fea_len,
                                                                       edge_dim=nbr_fea_len, heads=n_heads,
                                                                       concat=False)
                                          for _ in range(n_conv)])
        elif gcn1:
            self.n_convs = nn.ModuleList(
                [GCN_Layer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len, num_nbr=max_num_nbr)
                 for _ in range(n_conv)])
        else:
            self.n_convs_new = nn.ModuleList(
                [GCN_2(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len, num_nbr=max_num_nbr)
                 for _ in range(n_conv)])

        self.conv_to_fc = nn.Linear(atom_fea_len * (n_conv+1), h_fea_len)
        if l1 > 0:
            self.l1 = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                     for _ in range(l1)])

        if l2 > 0:
            self.l2 = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                     for _ in range(l2)])

        if self.p > 0:
            self.dropout = nn.Dropout(p=p)
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, n_classes)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data):
        x, edge_index, edge_weight, y = data.x, data.edge_index, data.edge_attr, data.y
        batch = data.batch
        x = self.atom_embedding(x)
        edge_weight = self.bond_embedding(edge_weight)

        temp = x

        if hasattr(self, 'n_convs'):
            for conv in self.n_convs:
                x = conv(x=x, edge_index=edge_index, edge_attr=edge_weight)
        if hasattr(self, 'n_convs_new'):
            for conv in self.n_convs_new:
                x, edge_weight = conv(x=x, edge_index=edge_index, edge_attr=edge_weight)
                temp = torch.cat((temp, x), dim=1)

        x = temp
        x = self.conv_to_fc(x)
        if hasattr(self, 'l1'):
            for hidden in self.l1:
                x = F.softplus(hidden(x))

        if self.pooling == 'add':
            x = torch_geometric.nn.global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = torch_geometric.nn.global_max_pool(x, batch)
        else:
            x = torch_geometric.nn.global_mean_pool(x, batch)

        if hasattr(self, 'l2'):
            for hidden in self.l2:
                x = F.softplus(hidden(x))
        out = self.fc_out(x)

        if self.p > 0:
            x = self.dropout(x)
        return out
