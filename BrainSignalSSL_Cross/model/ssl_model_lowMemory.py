import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import time
# import sys
# print(sys.path)
# sys.path.append('/home/caidonghong/Time-Series_SSL')

# from ..utils.data_preprocessing import predefined_graph
from .transformer_layer import BertModel
from utils.data_preprocessing import similarity_mean, similarity_mean_low_frequency, similarity_mean_large, similarity_mean_final
from .weighted_gcn import GCNConv
# from .plot_api import draw_heatmap


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=1):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        assert(len(hidden_dim) >= layer_num)
        for i in range(layer_num):
            if i == 0:
                setattr(self, 'conv' + str(i), GCNConv(input_dim, hidden_dim[i], add_self_loops=False))
            else:
                setattr(self, 'conv' + str(i), GCNConv(hidden_dim[i-1], hidden_dim[i], add_self_loops=False))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        all_hidden_representation = []
        for i in range(self.layer_num):
            # delete relu function in the last layer
            x = getattr(self, 'conv' + str(i))(x, edge_index, edge_weight)
            if i != self.layer_num - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=0.5)
            all_hidden_representation.append(x)
            # x = F.relu(getattr(self, 'conv' + str(i))(x, edge_index, edge_weight))
            # all_hidden_representation.append(x)
            # if i != self.layer_num - 1:
            #     x = F.dropout(x, p=0.5)
        return all_hidden_representation


class ChannelNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-05, affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cum_mean = x.mean(dim=1, keepdim=True)
        cum_var = x.var(dim=1, keepdim=True)
        x = (x - cum_mean) * torch.rsqrt(cum_var + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(CPCEncoder, self).__init__()
        norm_layer = ChannelNorm
        # 1000 HZ
        # self.conv1 = nn.Conv1d(1, hidden_size, 8, stride=4, padding=2)
        # self.batchNorm1 = norm_layer(hidden_size)
        # self.conv2 = nn.Conv1d(hidden_size, hidden_size, 4, stride=2, padding=1)
        # self.batchNorm2 = norm_layer(hidden_size)
        # self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, stride=2, padding=0)
        # self.batchNorm3 = norm_layer(hidden_size)
        # self.DOWNSAMPLING = 16

        # 250 HZ
        self.conv1 = nn.Conv1d(1, hidden_size, 4, stride=2, padding=1)
        self.batchNorm1 = norm_layer(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 4, stride=2, padding=1)
        self.batchNorm2 = norm_layer(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.batchNorm3 = norm_layer(hidden_size)
        self.DOWNSAMPLING = 4

    def get_dim_output(self):
        return self.conv3.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        return x


class CPCAR(nn.Module):
    def __init__(self,
                 dim_encoded,  # the dimension of input
                 dim_output,   # the dimension of output (hidden layer)
                 n_predicts,   # total steps for forecasting
                 keep_hidden,  # keep hidden layer's value during backend
                 nLevelsAR,    # stacking depth
                 mode='GRU'):
        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1
        if mode == "LSTM":
            self.baseNet = nn.LSTM(dim_encoded, dim_output,
                                   num_layers=nLevelsAR, batch_first=True, bidirectional=True)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dim_encoded, dim_output,
                                  num_layers=nLevelsAR, batch_first=True, bidirectional=True)
        elif mode == "GRU":
            self.baseNet = nn.GRU(dim_encoded, dim_output,
                                  num_layers=nLevelsAR, batch_first=True, bidirectional=True)
        else:
            # seq_len based on the sequence length after encoder
            self.baseNet = BertModel(
                input_size=dim_encoded,
                n_predicts=n_predicts,
                seq_len=62,
                hidden_size=dim_output,
                position_embedding_type='static'
            )
        self.ar_mode = mode
        self.hidden = None
        self.keepHidden = keep_hidden

    def get_dim_output(self):
        return self.baseNet.hidden_size

    def forward(self, x, mask_state):
        if self.ar_mode == 'TRANSFORMER':
            x = self.baseNet(
                x,
                mask_state=mask_state
            )
        else:
            try:
                self.baseNet.flatten_parameters()
            except RuntimeError:
                pass
            x, h = self.baseNet(x, self.hidden)  # (h_0, c_0) is None
            if self.keepHidden:
                if isinstance(h, tuple):
                    self.hidden = tuple(x.detach() for x in h)
                else:
                    self.hidden = h.detach()
        return x


class CPCModel(nn.Module):
    def __init__(self,
                 hidden_dim,            #
                 channel_num,           #
                 gcn_dim,               #
                 k,                     # KNN-Graph
                 n_predicts,            # total steps for forecasting
                 graph_construct,
                 direction,
                 args,
                 replace_ratio=0.15,    # replace ratio in replacement task
                 ar_mode='GRU',          #
                 ):
        super(CPCModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.channel_num = channel_num
        self.gcn_dim = gcn_dim
        self.k = k
        self.nPredicts = n_predicts
        self.graph_construct = graph_construct
        self.direction = direction
        self.replace_ratio = replace_ratio
        self.ar_mode = ar_mode

        # one Encoder and one Gar new_model for all channel
        self.gEncoder = CPCEncoder(hidden_size=hidden_dim)
        self.gAR = CPCAR(hidden_dim, hidden_dim, n_predicts, False, 1, mode=ar_mode)

        self.fGCN = GCN(hidden_dim, gcn_dim, 1)
        if self.direction == 'bi':
            self.bGCN = GCN(hidden_dim, gcn_dim, 1)


        self.W = nn.parameter.Parameter(torch.randn(size=(hidden_dim, hidden_dim)), requires_grad=True)
        self.cosSimilarity = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=0)
        # self.selfLoop_w = nn.parameter.Parameter(torch.tensor([0.5]), requires_grad=True)

        if graph_construct == 'gumbel':
            self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)
            self.linear_cat = nn.Linear(hidden_dim, 2)
            def encode_onehot(labels):
                classes = set(labels)
                classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
                labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
                return labels_onehot
            # Generate off-diagonal interaction graph
            off_diag = np.ones([channel_num, channel_num])
            rel_source = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            rel_target = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            self.rel_source = torch.FloatTensor(rel_source).cuda()
            self.rel_target = torch.FloatTensor(rel_target).cuda()
            self.mask_matrix = torch.ones((self.channel_num, self.channel_num)) - torch.eye(self.channel_num)
            self.mask_matrix = self.mask_matrix.cuda()
            self.threshold = 0.5

            # create a predefined weight matrix
            # self.predefined_matrix = predefined_graph(patient='02GJX', threshold=0.2)

        elif graph_construct == 'cos_threshold':
            self.mask_matrix = torch.ones((self.channel_num, self.channel_num)) - torch.eye(self.channel_num)
            self.mask_matrix = self.mask_matrix.cuda()
            self.threshold = 0.5
            self.adj_weight = 0.3

        elif graph_construct == 'sample_from_distribution':
            self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)
            self.linear_cat = nn.Linear(hidden_dim, 1)

            # def encode_onehot(labels):
            #     classes = set(labels)
            #     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            #     labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            #     return labels_onehot
            # # Generate off-diagonal interaction graph
            # off_diag = np.ones([channel_num, channel_num])
            # rel_source = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            # rel_target = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            # self.rel_source = torch.FloatTensor(rel_source).cuda()
            # self.rel_target = torch.FloatTensor(rel_target).cuda()

            # predefined mean matrix of similarity
            self.mean_matrix = similarity_mean_final(
                database_save_dir=args.database_save_dir,
                data_save_dir=args.data_save_dir,
            )
            self.mean_matrix = self.mean_matrix.cuda()

            self.threshold = args.graph_threshold
            self.adj_weight = args.adj_weight

            self.Softplus = torch.nn.Softplus()


    def sample_replace_pair(self, tot_num, seq_size):
        num1, num2 = 0, 0
        while (num1 // seq_size) == (num2 // seq_size):
            num1 = np.random.randint(low=0, high=tot_num)
            num2 = np.random.randint(low=0, high=tot_num)
        return num1, num2

    def sample_replace_data(self, encoded_data):
        batch_size, channel_num, seq_size, dim_en = encoded_data.size()
        tot_num = batch_size * channel_num * seq_size
        replace_num = int(tot_num * self.replace_ratio)
        source_idx = np.random.choice(tot_num, replace_num, replace=False)
        target_idx = np.random.choice(tot_num, replace_num, replace=False)

        neg_ext = encoded_data.contiguous().view(-1, dim_en)
        idx = np.arange(tot_num)
        idx[target_idx] = source_idx
        replace_data = neg_ext[idx].view(batch_size, channel_num, seq_size, dim_en)

        # 0 : not replaced || 1 : replaced
        if_replace = (source_idx // seq_size) != (target_idx // seq_size)
        replace_label = torch.zeros((tot_num), dtype=torch.long, device=encoded_data.device)
        replace_label[target_idx] = torch.tensor(if_replace, dtype=torch.long, device=encoded_data.device)

        return replace_data, replace_label



        # batch_size, channel_num, seq_size, dim_en = encoded_data.size()
        # tot_num = batch_size * channel_num * seq_size
        # replace_num = int(tot_num * self.replace_ratio * 0.5)
        # replace_label = torch.zeros((tot_num), dtype=torch.long, device=encoded_data.device)
        # # 0 : not replaced || 1 : replaced
        #
        # replace_pair = [[],[]]
        # first_num = np.random.randint(low=0, high=tot_num, size=(replace_num))
        # second_num = np.random.randint(low=0, high=tot_num, size=(replace_num))
        # first_num_seq = first_num // seq_size
        # second_num_seq = second_num // seq_size
        # for i in range(replace_num):
        #     if first_num_seq[i] != second_num_seq[i]:
        #         replace_pair[0].append(first_num[i])
        #         replace_pair[1].append(second_num[i])
        #     else:
        #         num1, num2 = self.sample_replace_pair(tot_num, seq_size)
        #         replace_pair[0].append(num1)
        #         replace_pair[1].append(num2)
        #
        # neg_ext = encoded_data.contiguous().view(-1, dim_en)
        # idx = [i for i in range(tot_num)]
        # for i in range(replace_num):
        #     idx[replace_pair[0][i]] = replace_pair[1][i]
        #     idx[replace_pair[1][i]] = replace_pair[0][i]
        #     replace_label[replace_pair[0][i]] = replace_label[replace_pair[1][i]] = 1
        #
        # replace_data = neg_ext[idx].view(batch_size, channel_num, seq_size, dim_en)

        # return replace_data, replace_label

    def sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape).to(device)
        # -log(-log(s + eps) + eps) ;where s is sampled from Uniform(0,1)
        return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature, eps=1e-10):
        sample = self.sample_gumbel(logits.size(), device=logits.device, eps=eps)
        y = logits + sample
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False, eps=1e-10):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y_soft = self.gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
        if hard:
            shape = logits.size()
            _, k = y_soft.data.max(-1)
            y_hard = torch.zeros(*shape).to(logits.device)
            y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
            y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
        else:
            y = y_soft
        return y

    def diffusion_module(self, after_gAR_batch, forward=True):
        ##################################################################
        ######################## Code Optimization #######################
        ##################################################################
        batch_size, node_num = after_gAR_batch.size()[:2]
        # after_gAR_batch.size(): (batch_size * time_span) * channel_num * dim_ar

        if self.graph_construct == 'sample_from_distribution':
            source_nodes = after_gAR_batch.view(batch_size, -1, 1, after_gAR_batch.size(-1)).expand(-1, -1, node_num, -1)
            target_nodes = after_gAR_batch.view(batch_size, 1, -1, after_gAR_batch.size(-1)).expand(-1, node_num, -1, -1)
            x = torch.cat([source_nodes, target_nodes], dim=-1)
            x = F.relu(self.linear_out(x))
            x = self.linear_cat(x)
            var_matrix = x.view(batch_size, node_num, node_num)

            normal_distribution = torch.randn_like(var_matrix, device=var_matrix.device)
            sample_weight = self.mean_matrix + self.Softplus(var_matrix) * normal_distribution

            mask_matrix = torch.ones((node_num, node_num)) - torch.eye(node_num)
            mask_matrix = mask_matrix.cuda()
            sample_weight = sample_weight * mask_matrix

            i, j, k = torch.where(sample_weight >= self.threshold)

            idx_shift = torch.tensor([batch_num * node_num for batch_num in range(batch_size)],
                                      dtype=torch.long, device=sample_weight.device)

            weights = sample_weight[i, j, k]
            j = j + idx_shift[i]
            k = k + idx_shift[i]

            edge_num = len(i)
            edge_weight = torch.sum(weights).item()

            edge_index = torch.stack([j, k], dim=0)
            graph = Data(x=after_gAR_batch.contiguous().view(-1, after_gAR_batch.size(-1)), edge_index=edge_index,
                         edge_attr=weights)

            if forward:
                n_predicts_representation = self.fGCN(graph)
            else:
                n_predicts_representation = self.bGCN(graph)

            after_gnn_batch = torch.stack(n_predicts_representation, dim=0)
            after_gnn_batch = after_gnn_batch.view(batch_size, 1, node_num, after_gnn_batch.size(-1))
            # after_gnn_batch.size(): batch_size * n_predicts * channel * hidden_dim

            # if batch_size < 30:
            #     if forward:
            #         print("Forward Direction:")
            #     else:
            #         print("Backward Direction:")
            #     if edge_num:
            #         print("Average number of edges: %d; Average weight of edges: %.2f" % (
            #         (edge_num / batch_size), (edge_weight / edge_num)))
            #     elif not edge_num:
            #         print("None edge!")

            return after_gnn_batch

        else:
            raise Exception("Other graph learning modules code have not been optimized!")


        # after_gnn_mini_batch = []
        #
        # batch_size, channel_num, dim_ar = after_gAR_batch.size()
        # # dim_ar == self.hidden_dim
        #
        # # nodeWise_correlation = torch.zeros(size=(channel_num, channel_num))
        # edge_num = 0
        # edge_weights = 0
        #
        # for seg_num in range(batch_size):
        #     weights = []
        #     source_vertex = []
        #     target_vertex = []
        #
        #     c_features = after_gAR_batch[seg_num]
        #     # c_features.size(): channel_num * dim_ar
        #
        #     if self.graph_construct == 'mi':
        #         # mutual-information similarity
        #         pairwise_similar = c_features @ self.W @ c_features.t()
        #         norm_2 = torch.norm(c_features, dim=1)
        #         source_nodes = torch.repeat_interleave(norm_2, norm_2.size(0), dim=0)
        #         target_nodes = norm_2.repeat(norm_2.size(0))
        #         normalize_matrix = (source_nodes * target_nodes).view(c_features.size(0), c_features.size(0))
        #         pairwise_similar = pairwise_similar / normalize_matrix
        #
        #     elif self.graph_construct == 'cos':
        #         # cosine-similarity
        #         source_nodes = torch.repeat_interleave(c_features, c_features.size(0), dim=0)
        #         target_nodes = c_features.repeat(c_features.size(0), 1)
        #         pairwise_similar = self.cosSimilarity(source_nodes, target_nodes)
        #         pairwise_similar = pairwise_similar.view(c_features.size(0), c_features.size(0))
        #
        #     elif self.graph_construct == 'gumbel':
        #         # self.rel_source = self.rel_source.cuda()
        #         # self.rel_target = self.rel_target.cuda()
        #         # self.mask_matrix = self.mask_matrix.cuda()
        #         source_nodes = torch.matmul(self.rel_source, c_features)
        #         target_nodes = torch.matmul(self.rel_target, c_features)
        #         x = torch.cat([source_nodes, target_nodes], dim=-1)
        #         x = torch.relu(self.linear_out(x))
        #         x = self.linear_cat(x)
        #
        #         adj = self.gumbel_softmax(x, temperature=0.5, hard=False)
        #         adj = adj[:, 0].clone().reshape(self.channel_num, -1)
        #         adj = adj * self.mask_matrix
        #         source_index, target_index = torch.where(adj >= self.threshold)
        #
        #         edge_index = torch.tensor([list(source_index), list(target_index)], dtype=torch.long, device=after_gAR_batch.device)
        #         weights = adj[source_index, target_index]
        #
        #         graph = Data(x=c_features, edge_index=edge_index, edge_attr=weights)
        #         n_predicts_representation = self.GCN(graph)
        #         after_gnn_mini_batch.append(torch.stack(n_predicts_representation, dim=0))
        #         continue
        #
        #     elif self.graph_construct == 'cos_threshold':
        #         # add weight to self-loop instead of neighbour nodes
        #         # source_nodes = torch.repeat_interleave(c_features, c_features.size(0), dim=0)
        #         # target_nodes = c_features.repeat(c_features.size(0), 1)
        #         # pairwise_similar = self.cosSimilarity(source_nodes, target_nodes)
        #         # pairwise_similar = pairwise_similar.view(c_features.size(0), c_features.size(0))
        #         # pairwise_similar = pairwise_similar * self.mask_matrix
        #         #
        #         # source_index, target_index = torch.where(pairwise_similar >= self.threshold)
        #         # weights = pairwise_similar[source_index, target_index]
        #         #
        #         # edge_num += len(source_index)
        #         # edge_weights += torch.sum(weights).item()
        #         #
        #         # # add softmax for weights of one node's neighbours
        #         # # softmax_weights = [self.selfLoop_w for _ in range(self.channel_num)]
        #         # softmax_weights = []
        #         # for idx in set(source_index.detach().cpu().numpy()):
        #         #     softmax_weights.append(self.selfLoop_w * self.softmax(weights[torch.where(source_index == idx)]))
        #         # weights = torch.cat(softmax_weights, dim=0)
        #         #
        #         # # self_index = torch.tensor([i for i in range(self.channel_num)], device=source_index.device)
        #         # # source_index = torch.cat((self_index, source_index))
        #         # # target_index = torch.cat((self_index, target_index))
        #         # edge_index = torch.stack([source_index, target_index], dim=0)
        #         #
        #         # graph = Data(x=c_features, edge_index=edge_index, edge_attr=weights)
        #         # if forward:
        #         #     n_predicts_representation = self.fGCN(graph)
        #         # else:
        #         #     n_predicts_representation = self.bGCN(graph)
        #         #
        #         # after_gnn_mini_batch.append(torch.stack(n_predicts_representation, dim=0))
        #         #
        #         # continue
        #
        #         source_nodes = torch.repeat_interleave(c_features, c_features.size(0), dim=0)
        #         target_nodes = c_features.repeat(c_features.size(0), 1)
        #         pairwise_similar = self.cosSimilarity(source_nodes, target_nodes)
        #         pairwise_similar = pairwise_similar.view(c_features.size(0), c_features.size(0))
        #         pairwise_similar = pairwise_similar * self.mask_matrix
        #
        #         source_index, target_index = torch.where(pairwise_similar >= self.threshold)
        #         edge_index = torch.stack([source_index, target_index], dim=0)
        #         weights = pairwise_similar[source_index, target_index]
        #
        #         edge_num += len(source_index)
        #         edge_weights += torch.sum(weights).item()
        #
        #         # add softmax for weights of one node's neighbours
        #         softmax_weights = []
        #         for idx in set(source_index.detach().cpu().numpy()):
        #             softmax_weights.append(self.adj_weight * self.softmax(weights[torch.where(source_index==idx)]))
        #         if len(softmax_weights):
        #             weights = torch.cat(softmax_weights, dim=0)
        #
        #         graph = Data(x=c_features, edge_index=edge_index, edge_attr=weights)
        #         if forward:
        #             n_predicts_representation = self.fGCN(graph)
        #         else:
        #             n_predicts_representation = self.bGCN(graph)
        #
        #         after_gnn_mini_batch.append(torch.stack(n_predicts_representation, dim=0))
        #
        #         continue
        #
        #     elif self.graph_construct == 'sample_from_distribution':
        #         source_nodes = torch.matmul(self.rel_source, c_features)
        #         target_nodes = torch.matmul(self.rel_target, c_features)
        #         x = torch.cat([source_nodes, target_nodes], dim=-1)
        #         x = F.relu(self.linear_out(x))
        #         x = self.linear_cat(x)
        #         var_matrix = x.view(self.channel_num, self.channel_num)
        #         # var_matrix = torch.abs(var_matrix)
        #
        #         normal_distribution = torch.randn_like(var_matrix, device=var_matrix.device)
        #         # normal_distribution = torch.normal(torch.zeros([self.channel_num, self.channel_num]), torch.ones([self.channel_num, self.channel_num])).cuda()
        #         # pairwise_similar = self.mean_matrix + (var_matrix * normal_distribution)
        #         pairwise_similar = self.mean_matrix + self.Softplus(var_matrix) * normal_distribution
        #         # pairwise_similar = self.mean_matrix + (var_matrix / 2).exp() * normal_distribution
        #
        #         # pairwise_similar = torch.clamp(pairwise_similar, min=-1, max=1)
        #         # remove self loop
        #         mask_matrix = torch.ones((self.channel_num, self.channel_num)) - torch.eye(self.channel_num)
        #         mask_matrix = mask_matrix.cuda()
        #         pairwise_similar = pairwise_similar * mask_matrix
        #
        #         source_index, target_index = torch.where(pairwise_similar >= self.threshold)
        #         edge_index = torch.stack([source_index, target_index], dim=0)
        #         weights = pairwise_similar[source_index, target_index]
        #
        #         edge_num += len(source_index)
        #         edge_weights += torch.sum(weights).item()
        #
        #         # add softmax for weights of one node's neighbours
        #         # softmax_weights = []
        #         # for idx in set(source_index.detach().cpu().numpy()):
        #         #     softmax_weights.append(self.adj_weight * self.softmax(weights[torch.where(source_index == idx)]))
        #         # if len(softmax_weights):
        #         #     weights = torch.cat(softmax_weights, dim=0)
        #
        #         graph = Data(x=c_features, edge_index=edge_index, edge_attr=weights)
        #
        #         if forward:
        #             n_predicts_representation = self.fGCN(graph)
        #         else:
        #             n_predicts_representation = self.bGCN(graph)
        #
        #         after_gnn_mini_batch.append(torch.stack(n_predicts_representation, dim=0))
        #
        #         continue
        #
        #
        #     # nodeWise_correlation += pairwise_similar.detach().cpu()
        #
        #     # construct a KNN graph based on the pair-wise similarity
        #     for i in range(channel_num):
        #         values, indices = torch.sort(pairwise_similar[i], descending=True)
        #         source_vertex.extend([i] * self.k)
        #         selected_values = []
        #         for idx in range(channel_num):
        #             if indices[idx] != i:
        #                 target_vertex.append(indices[idx])
        #                 selected_values.append(values[idx])
        #             if len(selected_values) == self.k:
        #                 break
        #         selected_values = torch.stack(selected_values, dim=0)
        #         weights.append(self.softmax(selected_values))
        #         # min-max normalization
        #         # weights.extend((values[:self.k] - values[:self.k].min()) / (values[:self.k].max() - values[:self.k].min()))
        #
        #     # draw_heatmap(data=pair_wise_mi.detach().cpu(), batch_idx=batch_idx, index=seg_num, save_path='./correlation_pictures/')
        #
        #     edge_index = torch.tensor([source_vertex, target_vertex], dtype=torch.long, device=after_gAR_batch.device)
        #     weights = torch.cat(weights, dim=0)
        #
        #     graph = Data(x=c_features, edge_index=edge_index, edge_attr=weights)
        #     n_predicts_representation = self.GCN(graph)
        #     after_gnn_mini_batch.append(torch.stack(n_predicts_representation, dim=0))
        #
        # # nodeWise_correlation /= batch_size
        # # nodeWise_correlation -= torch.eye(channel_num)
        # # draw_heatmap(nodeWise_correlation)
        # # if batch_size < 30:
        # #     if forward:
        # #         print("Forward Direction:")
        # #     else:
        # #         print("Backward Direction:")
        # #     if edge_num:
        # #         print("Average number of edges: %d; Average weight of edges: %.2f"%((edge_num/batch_size), (edge_weights / edge_num)))
        # #     elif not edge_num:
        # #         print("None edge!")
        #
        #     # print("Learnable weight of self loop is: %.5f"%self.selfLoop_w.item())
        #
        # # after_gnn_batch.size(): batch_size * n_predicts * channel * hidden_dim
        # after_gnn_batch = torch.stack(after_gnn_mini_batch, dim=0)
        # return after_gnn_batch

    def forward(self, batch_data, task='prediction', train_stage=True):
        # batch_data.size(): BatchSize * ChannelNum * SeqLength
        # graph_construct: graph construction method: ['cos', 'mi']
        # direction of AR new_model: ['single', 'bi']
        assert (task in ['prediction', 'replace', 'time-shift'])

        batch_size, channel_num, seq_len = batch_data.size()
        # batch_data = batch_data.contiguous().view(-1, 1, seq_len)
        batch_data = batch_data.view(-1, 1, seq_len)
        after_encoder = self.gEncoder(batch_data).permute(0, 2, 1)
        after_encoder_batch = after_encoder.view(batch_size, channel_num, after_encoder.size(-2), after_encoder.size(-1))

        if task == 'prediction':
            after_gAR = self.gAR(after_encoder, mask_state=self.direction)
            after_gAR_batch = after_gAR.view(batch_size, channel_num, after_gAR.size(-2), after_gAR.size(-1))

            ##################################################################
            ########################### GCN Module ###########################
            ##################################################################
            after_gnn_batch = []
            batch_size, channel_num, seq_size, dim_ar = after_gAR_batch.size()
            # if ar_mode != 'TRANSFORMER': dim_ar = hidden_dim * 2

            if train_stage:
                if self.direction == 'single':
                    window_size = seq_size - self.nPredicts
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, window_size - 1, :self.hidden_dim]))
                elif self.direction == 'bi':
                    # consider the forward direction
                    window_size = seq_size - (self.nPredicts // 2)
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, window_size - 1, :self.hidden_dim], True))
                    # consider the backward direction
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, self.nPredicts//2, -self.hidden_dim:], False))
            else:
                if self.graph_construct == 'NoGraph':
                    after_gnn_batch.append(after_gAR_batch[:, :, -2, :self.hidden_dim].unsqueeze(dim=1))
                elif self.direction == 'single':
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, -3, :self.hidden_dim]))
                elif self.direction == 'bi':
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, -3, :self.hidden_dim], True))
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, 2, -self.hidden_dim:], False))

            return after_encoder_batch, after_gAR_batch, after_gnn_batch

        elif task == 'replace':
            ##################################################################
            ################### Whether Replaced Prediction ##################
            ##################################################################
            replace_data, replace_label = self.sample_replace_data(after_encoder_batch)

            batch_data, channel_num, seq_len, enc_dim = replace_data.size()
            replace_data = replace_data.view(-1, seq_len, enc_dim)
            reaplce_after_gAR = self.gAR(replace_data, mask_state='single')
            reaplce_after_gAR = reaplce_after_gAR.contiguous().view(batch_size, channel_num, reaplce_after_gAR.size(-2), reaplce_after_gAR.size(-1))

            return reaplce_after_gAR, replace_label

        elif task == 'time-shift':
            after_gAR = self.gAR(after_encoder, mask_state=self.direction)
            after_gAR_batch = after_gAR.view(batch_size, channel_num, after_gAR.size(-2), after_gAR.size(-1))
            return after_gAR_batch
