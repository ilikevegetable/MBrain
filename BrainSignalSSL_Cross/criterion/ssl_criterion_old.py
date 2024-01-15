import torch
import torch.nn as nn
import torch.nn.functional as F

remove_last_steps = 0

class PredictionNetwork(nn.Module):
    def __init__(self,
                 n_predicts,
                 dim_output_gnn,
                 dim_output_encoder,
                 rnn_mode=None,
                 dropout=False):
        super(PredictionNetwork, self).__init__()
        self.nPredicts = n_predicts
        self.predictors = nn.ModuleList()  # the new_model used to predict c using z
        self.RESIDUAL_STD = 0.01
        self.dim_output_gnn = dim_output_gnn
        self.dropout = nn.Dropout(p=0.5) if dropout else None

        for i in range(n_predicts):
            if rnn_mode == 'RNN':
                self.predictors.append(nn.RNN(dim_output_gnn, dim_output_encoder))
                self.predictors[-1].flatten_parameters()
            elif rnn_mode == 'LSTM':
                self.predictors.append(nn.LSTM(dim_output_gnn, dim_output_encoder))
                self.predictors[-1].flatten_parameters()
            else:  # use an affine transformation to predict encoded features
                self.predictors.append(nn.Linear(dim_output_gnn * 2, dim_output_encoder, bias=False))
                if dim_output_encoder > dim_output_gnn:
                    residual = dim_output_encoder - dim_output_gnn
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(dim_output_gnn, dim_output_gnn),
                                                                     self.RESIDUAL_STD * torch.randn(residual, dim_output_gnn)], dim=0))

    def forward(self, c_forward, c_backward, candidates):
        # c_forward.size() = c_backward.size() : batch_size * n_predicts * dim_gnn
        # candidates.size(): [batch_size * (negativeSamplingExt + 1) * dim_encoded] * n_predicts
        # assert (len(candidates) == self.nPredicts)
        out = []

        for k in range(self.nPredicts - remove_last_steps):
            if c_backward == None:
                # we use one representation after gcn to predict nPredicts representation after ar model
                # so the k is all the same
                # locC = self.predictors[k](c_forward[:, k])
                locC = self.predictors[k](c_forward[:, 0])
            else:
                if k < self.nPredicts // 2:
                    # locC = self.predictors[k](c_forward[:, k])
                    locC = self.predictors[k](c_backward[:, 0])
                else:
                    # locC = self.predictors[k](c_backward[:, k - self.nPredicts//2])
                    locC = self.predictors[k](c_forward[:, 0])

            if isinstance(locC, tuple):
                locC = locC[0]
            if self.dropout is not None:
                locC = self.dropout(locC)

            locC = locC.view(locC.size(0), 1, locC.size(1))
            outK = (locC * candidates[k]).mean(dim=2)
            out.append(outK)

        return out  # a list, store k prediction of future c


class BaseCriterion(nn.Module):
    def warm_up(self):
        return False

    def update(self):
        return


class UnsupervisedCriterion(BaseCriterion):
    def __init__(self,
                 n_predicts,  # Number of steps
                 dim_output_gnn,  # Dimension of GNN
                 dim_output_encoder,  # Dimension of the convolution net
                 negative_sampling_ext,  # Number of negative samples to draw
                 direction,
                 rnn_mode=None,
                 dropout=False):
        super(UnsupervisedCriterion, self).__init__()

        self.wPrediction = PredictionNetwork(n_predicts, dim_output_gnn, dim_output_encoder, rnn_mode=rnn_mode, dropout=dropout)

        self.nPredicts = n_predicts
        self.negativeSamplingExt = negative_sampling_ext
        self.direction = direction
        self.lossCriterion = nn.CrossEntropyLoss()

    def sample_clean(self, encoded_data):
        # window_size is the input size of the AR new_model.
        # B*L*C: n_negative_ext is the encoded length;
        # dim_encoded is the length of features of the encoded data.
        batch_size, n_negative_ext, dim_encoded = encoded_data.size()
        outputs = []
        # Combine the batch and length in order to sample totally randomly.
        neg_ext = encoded_data.contiguous().view(-1, dim_encoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        # negative samples' index randomly
        batch_idx = torch.randint(low=0, high=batch_size,
                                  size=(self.negativeSamplingExt * batch_size,),
                                  device=encoded_data.device)

        # Draw the sequence index in every negative sample.
        seq_idx = torch.randint(low=1, high=n_negative_ext,
                                size=(self.negativeSamplingExt * batch_size,),
                                device=encoded_data.device)

        # base_idx = torch.arange(0, window_size, device=encoded_data.device)
        # base_idx = base_idx.view(1, 1, window_size).expand(1, self.negativeSamplingExt, window_size).expand(batch_size, self.negativeSamplingExt, window_size)

        # seq_idx += base_idx.contiguous().view(-1)
        # Keep the sequence index not out of the encoded length.
        # seq_idx = torch.remainder(seq_idx, n_negative_ext)

        ext_idx = seq_idx + batch_idx * n_negative_ext
        neg_ext = neg_ext[ext_idx].view(batch_size, self.negativeSamplingExt, dim_encoded)

        label_loss = torch.zeros((batch_size), dtype=torch.long, device=encoded_data.device)

        for k in range(self.nPredicts - remove_last_steps):
            if self.direction == 'bi':
                if k < self.nPredicts//2:
                    # pos_seq = encoded_data[:, k - self.nPredicts // 2]
                    pos_seq = encoded_data[:, k]
                else:
                    # pos_seq = encoded_data[:, self.nPredicts - 1 - k]
                    pos_seq = encoded_data[:, k - self.nPredicts]
            else:
                pos_seq = encoded_data[:, k - self.nPredicts]
            pos_seq = pos_seq.view(batch_size, 1, dim_encoded)
            full_seq = torch.cat((pos_seq, neg_ext), dim=1)
            outputs.append(full_seq)

        return outputs, label_loss

    def get_inner_loss(self):
        return "orthoLoss", self.orthoLoss * self.wPrediction.orthoCriterion()

    def forward(self, c_forward, c_backward, encoded_data):
        # c_forward.size() = c_backward().size(): batch_size * n_predicts * dim_gnn
        # encoded_data.size(): batch_size * seq_size * dim_encoder
        # direction: 'bi' or 'single'
        batch_size, _, dim_gnn = c_forward.size()
        # window_size = seq_size - self.nPredicts

        out_losses = [0 for _ in range(self.nPredicts - remove_last_steps)]
        out_acc = [0 for _ in range(self.nPredicts - remove_last_steps)]

        # sampled_data.size(): [batch_size * (negativeSamplingExt + 1) * dim_encoded] * n_predicts
        # label_loss.size(): batch_size // torch.zeros()
        sampled_data, label_loss = self.sample_clean(encoded_data)

        predictions = self.wPrediction(c_forward, c_backward, sampled_data)
        # predictions: a list, store k prediction of future c

        for k, loc_pred in enumerate(predictions):
            # loc_pred = loc_pred.permute(0, 2, 1)
            # loc_pred = loc_pred.contiguous().view(-1, loc_preds.size(1))

            lossK = self.lossCriterion(loc_pred, label_loss)
            out_losses[k] += lossK.view(1, -1)
            _, pred_index = loc_pred.max(1)
            out_acc[k] += torch.sum(pred_index == label_loss).float().view(1, -1)

            # if torch.sum(pred_index).item() > 0:
            #     print(torch.sum(pred_index).item())

        n_predicts_loss = torch.cat(out_losses, dim=1)
        n_predicts_acc = torch.cat(out_acc, dim=1) / batch_size

        return n_predicts_loss, n_predicts_acc


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=1):
        super(Discriminator, self).__init__()
        self.layer_num = layer_num
        self.predictors = nn.ModuleList()
        for i in range(layer_num):
            if i == 0:
                self.predictors.append(nn.Linear(input_dim, hidden_dim[0]))
            else:
                self.predictors.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        self.lossCriterion = nn.CrossEntropyLoss()

    def forward(self, data, label):
        data = data.view(-1, data.size(-1))
        label = label.view(-1,)
        for i in range(self.layer_num):
            data = self.predictors[i](data)
            if i != self.layer_num - 1:
                data = F.relu(data)
        loss = self.lossCriterion(data, label)
        acc = torch.eq(data.argmax(dim=1), label).sum() / label.size(0)

        return loss, acc


class ReplaceCriterion(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=1):
        super(ReplaceCriterion, self).__init__()
        self.layer_num = layer_num
        self.predictors = nn.ModuleList()
        for i in range(layer_num):
            if i == 0:
                self.predictors.append(nn.Linear(input_dim, hidden_dim[0]))
            else:
                self.predictors.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        self.lossCriterion = nn.CrossEntropyLoss()

    def forward(self, replace_data, replace_label):
        data = replace_data.view(-1, replace_data.size(-1))
        for i in range(self.layer_num):
            data = self.predictors[i](data)
            if i != self.layer_num - 1:
                data = F.relu(data)
        loss = self.lossCriterion(data, replace_label)
        acc = torch.eq(data.argmax(dim=1), replace_label).sum() / replace_label.size(0)

        return loss, acc


class TimeShiftCriterion(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=1):
        super(TimeShiftCriterion, self).__init__()
        self.layer_num = layer_num
        self.predictors = nn.ModuleList()
        for i in range(layer_num):
            if i == 0:
                self.predictors.append(nn.Linear(input_dim, hidden_dim[0]))
            else:
                self.predictors.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        self.lossCriterion = nn.CrossEntropyLoss()

    def forward(self, replace_data, replace_label):
        data = replace_data.view(-1, replace_data.size(-1))
        label = replace_label.view(-1,)
        for i in range(self.layer_num):
            data = self.predictors[i](data)
            if i != self.layer_num - 1:
                data = F.relu(data)
        loss = self.lossCriterion(data, label)
        acc = torch.eq(data.argmax(dim=1), label).sum() / label.size(0)

        return loss, acc