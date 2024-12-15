import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

remove_last_steps = 0


class PredictionNetwork(nn.Module):
    def __init__(self,
                 n_predicts,
                 dim_output_concat,
                 dim_output_encoder,
                 rnn_mode=None,
                 dropout=False):
        super(PredictionNetwork, self).__init__()
        self.nPredicts = n_predicts
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dim_output_concat = dim_output_concat
        self.dropout = nn.Dropout(p=0.5) if dropout else None

        for i in range(n_predicts):
            if rnn_mode == 'RNN':
                self.predictors.append(nn.RNN(dim_output_concat, dim_output_encoder))
                self.predictors[-1].flatten_parameters()
            elif rnn_mode == 'LSTM':
                self.predictors.append(nn.LSTM(dim_output_concat, dim_output_encoder))
                self.predictors[-1].flatten_parameters()
            else:
                # use an affine transformation to predict encoded features
                self.predictors.append(nn.Linear(dim_output_concat, dim_output_encoder, bias=False))
                if dim_output_encoder > dim_output_concat:
                    residual = dim_output_encoder - dim_output_concat
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(dim_output_concat, dim_output_concat),
                                                                     self.RESIDUAL_STD * torch.randn(residual, dim_output_concat)], dim=0))

    def forward(self, c_forward, c_backward, candidates):
        # c_forward.size() = c_backward.size() : batch_size * 1 * dim_concat
        # candidates.size(): batch_size * (negativeSamplingExt + 1) * dim_encoded
        out = []

        for k in range(self.nPredicts - remove_last_steps):
            if c_backward == None:
                locC = self.predictors[k](c_forward[:, 0])
            else:
                if k < self.nPredicts // 2:
                    locC = self.predictors[k](c_backward[:, 0])
                else:
                    locC = self.predictors[k](c_forward[:, 0])

            if isinstance(locC, tuple):
                locC = locC[0]
            if self.dropout is not None:
                locC = self.dropout(locC)

            locC = locC.view(locC.size(0), 1, locC.size(1))
            outK = (locC * candidates[k]).mean(dim=2)
            out.append(outK)

        # a list, store k prediction of future c
        return out


class UnsupervisedCriterion(nn.Module):
    def __init__(self,
                 n_predicts,  # Number of steps
                 dim_output_concat,  # Dimension of representation after concat
                 dim_output_encoder,  # Dimension of the convolution net
                 negative_sampling_ext,  # Number of negative samples to draw
                 direction,
                 rnn_mode=None,
                 dropout=False):
        super(UnsupervisedCriterion, self).__init__()

        self.wPrediction = PredictionNetwork(
            n_predicts=n_predicts,
            dim_output_concat=dim_output_concat,
            dim_output_encoder=dim_output_encoder,
            rnn_mode=rnn_mode,
            dropout=dropout,
        )

        self.nPredicts = n_predicts
        self.negativeSamplingExt = negative_sampling_ext
        self.direction = direction
        self.lossCriterion = nn.CrossEntropyLoss()

    def sample_clean(self, encoded_data):
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
        seq_idx = torch.randint(low=0, high=n_negative_ext,
                                size=(self.negativeSamplingExt * batch_size,),
                                device=encoded_data.device)

        ext_idx = seq_idx + batch_idx * n_negative_ext
        neg_ext = neg_ext[ext_idx].view(batch_size, self.negativeSamplingExt, dim_encoded)

        label = torch.zeros((batch_size), dtype=torch.long, device=encoded_data.device)

        for k in range(self.nPredicts - remove_last_steps):
            if self.direction == 'bi':
                if k < self.nPredicts//2:
                    pos_seq = encoded_data[:, k]
                else:
                    pos_seq = encoded_data[:, k - self.nPredicts]
            else:
                pos_seq = encoded_data[:, k - self.nPredicts]
            pos_seq = pos_seq.view(batch_size, 1, dim_encoded)
            full_seq = torch.cat((pos_seq, neg_ext), dim=1)
            outputs.append(full_seq)

        return outputs, label

    def forward(self, c_forward, c_backward, encoded_data):
        # c_forward.size(): batch_size * 1 * channel_num * dim_concat
        # encoded_data.size(): batch_size * channel_num * seq_size * dim_enc
        c_forward = c_forward.permute(0,2,1,3)
        c_forward = c_forward.view(-1, c_forward.size(2), c_forward.size(3))
        encoded_data = encoded_data.view(-1, encoded_data.size(2), encoded_data.size(3))
        if c_backward:
            c_backward = c_backward.permute(0,2,1,3)
            c_backward = c_backward.view(-1, c_backward.size(2), c_backward.size(3))

        batch_size = c_forward.size()[0]

        out_losses = [0 for _ in range(self.nPredicts - remove_last_steps)]
        out_acc = [0 for _ in range(self.nPredicts - remove_last_steps)]

        # sampled_data.size(): batch_size * (negativeSamplingExt + 1) * dim_enc
        sampled_data, label = self.sample_clean(encoded_data)

        predictions = self.wPrediction(c_forward, c_backward, sampled_data)
        # predictions: a list, store k prediction of future c

        for k, loc_pred in enumerate(predictions):
            lossK = self.lossCriterion(loc_pred, label)
            out_losses[k] += lossK.view(1, -1)
            # out_losses[k].size(): (1,1)
            _, pred_index = loc_pred.max(1)
            out_acc[k] += torch.sum(pred_index == label).float().view(1, -1)

        n_predicts_loss = torch.cat(out_losses, dim=1)
        n_predicts_acc = torch.cat(out_acc, dim=1) / batch_size
        # n_predicts_loss.size(): (1,8)
        # n_predicts_acc.size(): (1,8)

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


def time_shift_measurement(
        x,
        time_shift_score,
        time_shift_method,
        measure_steps,
        sample_ratio,
        time_shift_threshold):
    # x.size(): batch_size * time_span * channel_num * dim_concat

    batch_size, time_span, channel_num, dim_concat = x.size()

    if time_shift_method == 'sample_idx':
        all_num = measure_steps * channel_num
        random_list = [np.random.choice(all_num, int(all_num * sample_ratio), replace=False) for _ in
                       range(batch_size)]

        all_source = []
        all_target = []
        all_score = []
        for batch_idx in range(batch_size):
            batch_data = x[batch_idx]
            source_batch = []
            target_batch = []
            score_batch = []
            for i in range(time_span - measure_steps):
                source = torch.repeat_interleave(batch_data[i], int(measure_steps * channel_num * sample_ratio), dim=0)
                target = batch_data[i + 1:i + 1 + measure_steps].view(-1, dim_concat)[random_list[batch_idx]].repeat(channel_num, 1)
                source_batch.append(source)
                target_batch.append(target)

                sample_score = [time_shift_score[batch_idx,i][random_list[batch_idx] + c*all_num] for c in range(channel_num)]
                sample_score = torch.cat(sample_score)
                score_batch.append(sample_score)

            all_source.append(torch.stack(source_batch, dim=0))
            all_target.append(torch.stack(target_batch, dim=0))
            all_score.append(torch.stack(score_batch, dim=0))

        all_score = torch.stack(all_score, dim=0)
        all_source = torch.stack(all_source, dim=0)
        all_target = torch.stack(all_target, dim=0)
        # all_source.size() = all_target.size(): batch_size * (time_span - measure_steps) * (channel_num**2 * measure_step * sample_ratio) * dim_concat
        concat_rep = torch.cat((all_source, all_target), dim=-1)

        x_idx, y_idx, z_idx = torch.where(all_score >= time_shift_threshold)

        timeShift_label = torch.zeros_like(all_score, dtype=torch.long, device=x.device)
        # 0 : not shifted || 1 : shifted
        timeShift_label[x_idx, y_idx, z_idx] = 1

        return concat_rep, timeShift_label, (len(x_idx)/int(batch_size * channel_num**2 * measure_steps * (time_span - measure_steps) * sample_ratio))
