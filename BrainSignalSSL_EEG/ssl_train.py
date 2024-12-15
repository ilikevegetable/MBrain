import torch
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
import time
import sys
import os

sys.path.append('criterion')
sys.path.append('model')
sys.path.append('pipelines4EEG')
sys.path.append('utils')

import utils.misc as utilss
from model.ssl_model import MBrain
from criterion.ssl_criterion import UnsupervisedCriterion, Discriminator, time_shift_measurement
from pipelines4EEG.dataloader_ssl import load_dataset_ssl


def train_step(data_loader,
               ssl_model,
               ssl_criterion,
               replace_criterion,
               timeShift_criterion,
               optimizer,
               args,
               epoch):

    ssl_model.train()
    ssl_criterion.train()
    replace_criterion.train()
    timeShift_criterion.train()

    n_examples = 0
    logs, last_logs = {}, None
    iter_count = 0
    batch_tot_loss = 0
    time_shift_ratio = 0
    replace_loss, replace_acc = torch.zeros(1), torch.zeros(1)
    timeShift_loss, timeShift_acc = torch.zeros(1), torch.zeros(1)

    for step, full_data in enumerate(tqdm(data_loader, disable=args.tqdm_disable)):
        batch_data, time_shift_score = full_data
        n_examples += batch_data.size(0)
        batch_data = batch_data.cuda(non_blocking=True)
        # batch_data.size(): batch_size * 10 * channel_num * dim
        # time_shift_score.size(): batch_size * 4 * (idx_num)

        batch_size, time_span, _, _ = batch_data.size()
        batch_data = batch_data.view(batch_size * time_span, batch_data.size(2), batch_data.size(-1))

        ##################################################################
        ######################### Prediction Task ########################
        ##################################################################
        after_encoder, after_gAR, after_gnn, replace_after_gAR, replace_label = ssl_model(batch_data, train_stage=True)
        # after_gAR.size(): batch_size * channel_num * seq_size * dim_ar
        # after_gnn is a list, len(after_gnn) == 2 if mask_state == 'bi'
        # after_gnn.size(): batch_size * n_predicts * channel * hidden_dim
        tot_loss = 0

        # Concat After_AR and After_GNN (without self-loop) representation.
        if args.direction == 'bi':
            window_size = after_gAR.size(2) - (args.n_predicts // 2)
            gAR_f = after_gAR[:, :, window_size - 1]
            gAR_f = gAR_f.reshape(gAR_f.size(0), 1, gAR_f.size(1), gAR_f.size(2))
            # consider the backward direction
            gAR_b = after_gAR[:, :, args.n_predicts // 2]
            gAR_b = gAR_b.reshape(gAR_b.size(0), 1, gAR_b.size(1), gAR_b.size(2))

            concat_f = torch.cat((gAR_f, after_gnn[0]), dim=-1)
            concat_b = torch.cat((gAR_b, after_gnn[1]), dim=-1)
        elif args.direction == 'single':
            window_size = after_gAR.size(2) - args.n_predicts
            gAR_f = after_gAR[:, :, window_size-1, :args.hidden_dim]
            gAR_f = gAR_f.reshape(gAR_f.size(0), 1, gAR_f.size(1), gAR_f.size(2))
            concat_f = torch.cat((gAR_f, after_gnn[0]), dim=-1)
            concat_b = None
        else:
            raise Exception("Wrong direction!")

        pred_loss, pred_acc = ssl_criterion(concat_f, concat_b, after_encoder)
        pred_losses = pred_loss.detach().cpu()
        pred_acc = pred_acc.cpu()

        if epoch >= args.start_rep and epoch >= args.start_ts:
            tot_loss += pred_loss.sum() * (1 - args.lambda1 - args.lambda2)
        elif epoch >= args.start_rep and epoch < args.start_ts:
            tot_loss += pred_loss.sum() * (1 - args.lambda1)
        elif epoch < args.start_rep and epoch >= args.start_ts:
            tot_loss += pred_loss.sum() * (1 - args.lambda2)
        else:
            tot_loss = pred_loss.sum()


        ##################################################################
        ########################## Replace Task ##########################
        ##################################################################
        if epoch >= args.start_rep:
            replace_loss, replace_acc = replace_criterion(replace_after_gAR, replace_label)
            tot_loss += replace_loss.sum() * args.lambda1


        ##################################################################
        ######################### Time-shift Task ########################
        ##################################################################
        if epoch >= args.start_ts:
            after_gAR = after_gAR.view(batch_size, time_span, after_gAR.size(1), after_gAR.size(2), after_gAR.size(-1))
            # after_gAR.size(): batch_size * time_span * channel_num * seq_size * dim_ar
            # pooling in order to get 1-second representation
            # only consider single direction situation
            # all_steps
            r_max = torch.max(after_gAR[:, :, :, :, :args.hidden_dim], dim=3)[0]
            r_sum = torch.sum(after_gAR[:, :, :, :, :args.hidden_dim], dim=3)
            r_mean = torch.mean(after_gAR[:, :, :, :, :args.hidden_dim], dim=3)

            concat_representation = torch.cat((r_max, r_sum, r_mean), dim=-1)
            # concat_representation.size(): batch_size * time_span * channel_num * (dim_ar * 3)
            timeShift_rep, timeShift_label, timeShift_ratio = time_shift_measurement(
                x=concat_representation,
                time_shift_score=time_shift_score,
                time_shift_method=args.time_shift_method,
                measure_steps=args.measure_steps,
                sample_ratio=args.sample_ratio,
                time_shift_threshold=args.time_shift_threshold
            )
            time_shift_ratio += timeShift_ratio
            timeShift_loss, timeShift_acc = timeShift_criterion(timeShift_rep, timeShift_label)

            tot_loss += timeShift_loss.sum() * args.lambda2


        tot_loss.backward()
        batch_tot_loss += tot_loss.detach().cpu().numpy()

        if epoch >= args.start_rep:
            replace_loss = replace_loss.detach().cpu()
            replace_acc = replace_acc.cpu()
        if epoch >= args.start_ts:
            timeShift_loss = timeShift_loss.detach().cpu()
            timeShift_acc = timeShift_acc.cpu()


        if "predLoss_train" not in logs:
            logs["predLoss_train"] = np.zeros(pred_losses.size(1))
            logs["predAcc_train"] = np.zeros(pred_acc.size(1))
            logs["repLoss_train"] = np.zeros(1)
            logs["repAcc_train"] = np.zeros(1)
            logs["tsLoss_train"] = np.zeros(1)
            logs["tsAcc_train"] = np.zeros(1)

        iter_count += 1
        logs["predLoss_train"] += pred_losses.mean(dim=0).numpy()
        logs["predAcc_train"] += pred_acc.mean(dim=0).numpy()
        logs["repLoss_train"] += replace_loss.mean(dim=0).numpy()
        logs["repAcc_train"] += replace_acc.mean(dim=0).numpy()
        logs["tsLoss_train"] += timeShift_loss.mean(dim=0).numpy()
        logs["tsAcc_train"] += timeShift_acc.mean(dim=0).numpy()

        batch_num_to_step = args.batch_num_to_step
        accumulation_steps = batch_num_to_step//args.batch_size if batch_num_to_step > args.batch_size else 1
        if ((step+1)%accumulation_steps) == 0 or (step+1) == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()

    logs = utilss.update_logs(logs, iter_count)
    utilss.show_logs("Average Training Results:", logs)

    if epoch >= args.start_ts and epoch%5 == 0:
        print("The ratio of sampled edges in time-shift task:%.2f"%(time_shift_ratio/iter_count))

    return logs, batch_tot_loss


def val_step(data_loader,
             ssl_model,
             ssl_criterion,
             replace_criterion,
             timeShift_criterion,
             args,
             epoch):

    ssl_model.eval()
    ssl_criterion.eval()
    replace_criterion.eval()
    timeShift_criterion.eval()

    logs = {}
    iter_count = 0
    time_shift_ratio = 0
    replace_loss, replace_acc = torch.zeros(1), torch.zeros(1)
    timeShift_loss, timeShift_acc = torch.zeros(1), torch.zeros(1)

    for step, full_data in enumerate(tqdm(data_loader, disable=args.tqdm_disable)):
        batch_data, time_shift_score = full_data
        batch_data = batch_data.cuda(non_blocking=True)

        with torch.no_grad():
            batch_size, time_span, _, _ = batch_data.size()
            batch_data = batch_data.view(batch_size * time_span, batch_data.size(2), batch_data.size(-1))

            ##################################################################
            ######################### Prediction Task ########################
            ##################################################################
            after_encoder, after_gAR, after_gnn, replace_after_gAR, replace_label = ssl_model(batch_data, train_stage=True)

            # Concat After_AR and After_GNN (without self-loop) representation.
            if args.direction == 'bi':
                window_size = after_gAR.size(2) - (args.n_predicts // 2)
                gAR_f = after_gAR[:, :, window_size - 1]
                gAR_f = gAR_f.reshape(gAR_f.size(0), 1, gAR_f.size(1), gAR_f.size(2))
                # consider the backward direction
                gAR_b = after_gAR[:, :, args.n_predicts // 2]
                gAR_b = gAR_b.reshape(gAR_b.size(0), 1, gAR_b.size(1), gAR_b.size(2))

                concat_f = torch.cat((gAR_f, after_gnn[0]), dim=-1)
                concat_b = torch.cat((gAR_b, after_gnn[1]), dim=-1)
            elif args.direction == 'single':
                window_size = after_gAR.size(2) - args.n_predicts
                gAR_f = after_gAR[:, :, window_size - 1, :args.hidden_dim]
                gAR_f = gAR_f.reshape(gAR_f.size(0), 1, gAR_f.size(1), gAR_f.size(2))
                concat_f = torch.cat((gAR_f, after_gnn[0]), dim=-1)
                concat_b = None
            else:
                raise Exception("Wrong direction!")

            pred_loss, pred_acc = ssl_criterion(concat_f, concat_b, after_encoder)
            pred_losses = pred_loss.cpu()
            pred_acc = pred_acc.cpu()


            ##################################################################
            ########################## Replace Task ##########################
            ##################################################################
            if epoch >= args.start_rep:
                replace_loss, replace_acc = replace_criterion(replace_after_gAR, replace_label)
                replace_loss = replace_loss.cpu()
                replace_acc = replace_acc.cpu()


            ##################################################################
            ######################### Time-shift Task ########################
            ##################################################################
            if epoch >= args.start_ts:
                after_gAR = after_gAR.view(batch_size, time_span, after_gAR.size(1), after_gAR.size(2), after_gAR.size(-1))
                # after_gAR.size(): batch_size * time_span * channel_num * seq_size * dim_ar
                # pooling in order to get 1-second representation
                # only consider single direction situation
                # all_steps
                r_max = torch.max(after_gAR[:, :, :, :, :args.hidden_dim], dim=3)[0]
                r_sum = torch.sum(after_gAR[:, :, :, :, :args.hidden_dim], dim=3)
                r_mean = torch.mean(after_gAR[:, :, :, :, :args.hidden_dim], dim=3)

                concat_representation = torch.cat((r_max, r_sum, r_mean), dim=-1)
                # concat_representation.size(): batch_size * time_span * channel_num * (dim_ar * 3)
                timeShift_rep, timeShift_label, timeShift_ratio = time_shift_measurement(
                    x=concat_representation,
                    time_shift_score=time_shift_score,
                    time_shift_method=args.time_shift_method,
                    measure_steps=args.measure_steps,
                    sample_ratio=args.sample_ratio,
                    time_shift_threshold=args.time_shift_threshold
                )
                time_shift_ratio += timeShift_ratio
                timeShift_loss, timeShift_acc = timeShift_criterion(timeShift_rep, timeShift_label)
                timeShift_loss = timeShift_loss.cpu()
                timeShift_acc = timeShift_acc.cpu()


        if "predLoss_val" not in logs:
            logs["predLoss_val"] = np.zeros(pred_losses.size(1))
            logs["predAcc_val"] = np.zeros(pred_acc.size(1))
            logs["repLoss_val"] = np.zeros(1)
            logs["repAcc_val"] = np.zeros(1)
            logs["tsLoss_val"] = np.zeros(1)
            logs["tsAcc_val"] = np.zeros(1)

        iter_count += 1
        logs["predLoss_val"] += pred_losses.mean(dim=0).numpy()
        logs["predAcc_val"] += pred_acc.mean(dim=0).numpy()
        logs["repLoss_val"] += replace_loss.mean(dim=0).numpy()
        logs["repAcc_val"] += replace_acc.mean(dim=0).numpy()
        logs["tsLoss_val"] += timeShift_loss.mean(dim=0).numpy()
        logs["tsAcc_val"] += timeShift_acc.mean(dim=0).numpy()


    logs = utilss.update_logs(logs, iter_count)
    utilss.show_logs("Average Validation Results:", logs)

    if epoch >= args.start_ts and epoch%5 == 0:
        print("The ratio of sampled edges in time-shift task:%.2f"%(time_shift_ratio/iter_count))

    return logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training SSL model.')
    # ssl-model mode setting
    parser.add_argument('--ar_mode', type=str, default='LSTM',
                        help="The used AR model, including ['RNN', 'LSTM', 'GRU', 'TRANSFORMER']")
    parser.add_argument('--direction', type=str, default='single',
                        help="The direction for prediction task, including ['single', 'bi', 'no']")
    parser.add_argument('--graph_construct', type=str, default='sample_from_distribution',
                        help="The method for graph construction, including ['sample_from_distribution', 'predefined_distance']")
    parser.add_argument('--graph_threshold', type=float, default=0.5,
                        help='The threshold to sample edges in graph construct module.')
    # hyper parameters setting
    parser.add_argument('--n_predicts', type=int, default=8,
                        help='Number of time steps in prediction task.')
    parser.add_argument('--replace_ratio', type=float, default=0.15,
                        help='The ratio for replacing timestamps in replacement task.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The hidden dimension of model.')
    parser.add_argument('--negative_samples', type=int, default=128,
                        help='The number of negative samples in prediction task.')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[4, 4, 3],
                        help='The kernel size list for encoder (Conv1d).')
    parser.add_argument('--stride_size', type=int, nargs='+', default=[2, 2, 1],
                        help='The stride size list for encoder (Conv1d).')
    parser.add_argument('--padding_size', type=int, nargs='+', default=[0, 0, 0],
                        help='The padding size list for encoder (Conv1d).')
    # time-shift task setting
    parser.add_argument('--time_shift_method', type=str, default='sample_idx',
                        help='The way for time-shift measurement.')
    parser.add_argument('--measure_steps', type=int, default=7,
                        help='The number of steps to measure in time-shift task.')
    parser.add_argument('--time_shift_threshold', type=float, default=0.5,
                        help='The threshold to measure time-shift')
    parser.add_argument('--sample_ratio', type=float, default=0.5,
                        help='The ratio of negative sample for time-shift task.')
    # training details setting
    parser.add_argument('--start_rep', type=int, default=20,
                        help='The epoch to start replace task.')
    parser.add_argument('--start_ts', type=int, default=30,
                        help='The epoch to start time-shift task.')
    parser.add_argument('--lambda1', type=float, default=0.3,
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='The weight hyperparameter for the loss of replace discriminative task.')
    parser.add_argument('--lambda2', type=float, default=0.3,
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='The weight hyperparameter for the loss of delay time-shift predictive task.')
    parser.add_argument('--gpu', action='store_false',
                        help='Whether to use gpu.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size of input data.')
    parser.add_argument('--epochs', type=int, default=250,
                        help='The total number of epochs in the training stage.')
    parser.add_argument('--save_step', type=int, default=10,
                        help='The number of steps to save the checkpoint.')
    parser.add_argument('--early_stopping_epochs', type=int, default=10,
                        help='The number of epochs to stop training.')
    parser.add_argument('--batch_num_to_step', type=int, default=8,
                        help='The number of batches to step optimizer due to gpu-memory lack.')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate of SSL model in the training stage.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay of optimizer.')
    parser.add_argument('--tqdm_disable', action='store_true',
                        help='Whether disable the bar of tqdm module.')
    parser.add_argument('--save_dir', type=str, default='/data/EEG_checkpoints/',
                        help='The path for saving checkpoint.')
    parser.add_argument('--data_dir', type=str, default='/data/EEG_database/',
                        help='The path for saving EEG dataset.')
    args = parser.parse_args()

    num_threads = '32'
    torch.set_num_threads(int(num_threads))
    os.environ['OMP_NUM_THREADS'] = num_threads
    os.environ['OPENBLAS_NUM_THREADS'] = num_threads
    os.environ['MKL_NUM_THREADS'] = num_threads
    os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = num_threads

    print('-' * 20, 'print parameters', '-' * 20)
    for arg in vars(args):
        print(format(arg, '<25'), format(str(getattr(args, arg)), '<'))  # str, arg_type

    # torch.cuda.set_device(args.cuda_idx)

    # random_seed = random.randint(0, 2**31)
    print('\n', '-' * 20, 'loading dataset', '-' * 20)
    random_seed = 10
    utilss.set_seed(random_seed)
    print("Random seed:", random_seed)

    logs = {"epoch": [], "iter": [], "saveStep": args.save_step, "logging_step": 1000}

    dataloaders, _ = load_dataset_ssl(
        data_dir=args.data_dir,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=0,
    )

    ssl_model = MBrain(
        hidden_dim=args.hidden_dim,
        channel_num=19,
        gcn_dim=[256],
        n_predicts=args.n_predicts,
        graph_construct=args.graph_construct,
        direction=args.direction,
        replace_ratio=args.replace_ratio,
        ar_mode=args.ar_mode,
        args = args,
    )

    ssl_criterion = UnsupervisedCriterion(
        n_predicts=args.n_predicts,
        dim_output_concat=256*2,
        dim_output_encoder=args.hidden_dim,
        negative_sampling_ext=args.negative_samples,
        direction=args.direction,
        rnn_mode='linear',
        dropout=False
    )

    replace_criterion = Discriminator(
        input_dim=args.hidden_dim,
        hidden_dim=[128, 2],
        layer_num=2
    )

    timeShift_criterion = Discriminator(
        input_dim=args.hidden_dim*6,
        hidden_dim=[512, 2],
        layer_num=2
    )

    ssl_model.cuda()
    ssl_criterion.cuda()
    replace_criterion.cuda()
    timeShift_criterion.cuda()

    g_params = list(ssl_model.parameters()) + list(ssl_criterion.parameters()) + \
               list(replace_criterion.parameters()) + list(timeShift_criterion.parameters())

    optimizer = torch.optim.Adam(g_params, lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=args.weight_decay)

    # Checkpoint
    path_checkpoint = str(args.save_dir)
    if path_checkpoint is not None:
        if not os.path.isdir(path_checkpoint):
            os.mkdir(path_checkpoint)
        path_checkpoint = os.path.join(path_checkpoint, "checkpoint")

    print('\n', '-' * 20, 'start training ssl model', '-' * 20)
    # start_epoch = len(logs["epoch"])
    n_epoch = args.epochs
    start_epoch = 0
    acc_decrease_epoch = 0
    last_loss = 0
    best_acc = 0
    best_model_state = None
    start_time = time.time()
    training_ending_signal = False

    train_loader = dataloaders['train']
    val_loader = dataloaders['dev']
    print("Training Dataset: %d batches, Validation Dataset: %d batches, Batch Size: %d" %
          (len(train_loader), len(val_loader), args.batch_size))

    for epoch in range(start_epoch, n_epoch):
        print(f"\nStarting epoch {epoch + 1}")

        loc_logs_train, current_loss = \
            train_step(train_loader, ssl_model, ssl_criterion, replace_criterion, timeShift_criterion,
                       optimizer, args, epoch)

        loc_logs_val = \
            val_step(val_loader, ssl_model, ssl_criterion, replace_criterion, timeShift_criterion,
                     args, epoch)

        print(f'Ran {epoch + 1} epochs '
              f'in {time.time() - start_time:.2f} seconds')

        loss_change = np.fabs(current_loss - last_loss)
        last_loss = current_loss
        current_acc = float(1.0 * loc_logs_val["predAcc_val"].mean() + \
                            0.3 * loc_logs_val["repAcc_val"].mean() + \
                            0.3 * loc_logs_val["tsAcc_val"].mean())
        acc_decrease_epoch += 1

        if current_acc > best_acc:
            best_model_state = deepcopy(ssl_model.state_dict())
            best_acc = current_acc
            acc_decrease_epoch = 0
            print('Saved current model!')

        for key, value in dict(loc_logs_train, **loc_logs_val).items():
            if key not in logs:
                logs[key] = [None for _ in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch)

        if loss_change < 0.001 or acc_decrease_epoch >= args.early_stopping_epochs:
            training_ending_signal = True

        if path_checkpoint is not None \
                and ((epoch + 1) % logs["saveStep"] == 0 or epoch == n_epoch - 1 or training_ending_signal):
            ssl_state = ssl_model.state_dict()
            ssl_criterion_state = ssl_criterion.state_dict()
            replace_criterion_state = replace_criterion.state_dict()
            timeShift_criterion_state = timeShift_criterion.state_dict()
            optimizer_state = optimizer.state_dict()

            utilss.save_checkpoint_ts(ssl_state, ssl_criterion_state, replace_criterion_state,
                                     timeShift_criterion_state, optimizer_state, best_model_state,
                                     best_acc, f"{path_checkpoint}_{epoch + 1}.pt")
            utilss.save_logs(logs, path_checkpoint + "_logs.json")
            print('Logs updated!')

        if training_ending_signal:
            break

    print("After %d epochs training, ending!"%(epoch + 1))