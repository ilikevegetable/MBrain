import torch
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
import time
import sys
import os
import shutil
import warnings
import random

sys.path.append('criterion')
sys.path.append('model')
sys.path.append('pipelines4EEG')
sys.path.append('utils')

import utils.misc as utilss
from model.ssl_model import MBrain
from criterion.downstream_task_criterion import LinearClassifier4Emotion, DownstreamCriterion
from pipelines4EEG.dataloader_detection import  load_dataset_detection

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, fbeta_score, roc_auc_score
warnings.filterwarnings('ignore')


def train_step(data_loader,
               ssl_model,
               downstream_classifier,
               downstream_model,
               optimizer,
               args):

    ssl_model.train()
    downstream_classifier.train()
    downstream_model.train()
    optimizer.zero_grad()

    batch_tot_loss = 0
    logs, last_logs = {}, None
    iter_count = 0
    all_true_label = []
    all_pred_label = []

    for step, full_data in enumerate(tqdm(data_loader, disable=args.tqdm_disable)):
        batch_data, label = full_data
        # batch_data, label, brain_label, patient_label = full_data
        # batch_data.size(): batch_size * 10 * channel_num * window_length
        batch_data = batch_data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        batch_representation = []
        for batch_idx in range(batch_data.size(0)):
            _, after_gAR, _ = ssl_model(batch_data[batch_idx], train_stage=False)
            # after_gAR.size(): time_span * channel_num * seq_size * dim_ar

            if args.direction == 'single':
                # pooling
                r_max = torch.max(after_gAR[:, :, :, :args.hidden_dim], dim=2)[0]
                r_sum = torch.sum(after_gAR[:, :, :, :args.hidden_dim], dim=2)
                r_mean = torch.mean(after_gAR[:, :, :, :args.hidden_dim], dim=2)

                concat_representation = torch.cat((r_max, r_sum, r_mean), dim=-1)
                after_downAR = downstream_model(concat_representation)
                batch_representation.append(after_downAR)

            else:
                raise Exception("Bi-direction code have not been optimized!")

        batch_representation = torch.stack(batch_representation, dim=0)
        all_losses, all_acc, true_label_batch, pred_label_batch = downstream_classifier(batch_representation, label, True)
        tot_loss = all_losses.sum() / (args.batch_num_to_step // args.batch_size)

        tot_loss.backward()
        batch_tot_loss += tot_loss.detach().cpu().numpy()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(1)
            logs["locAcc_train"] = np.zeros(1)

        iter_count += 1
        logs["locLoss_train"] += (all_losses).detach().cpu().numpy()
        logs["locAcc_train"] += (all_acc).cpu().numpy()

        batch_num_to_step = args.batch_num_to_step
        accumulation_steps = batch_num_to_step // args.batch_size if batch_num_to_step > args.batch_size else 1
        if ((step+1) % accumulation_steps) == 0 or (step+1) == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()

        all_true_label.append(true_label_batch.view(-1).detach().cpu())
        all_pred_label.append(pred_label_batch.view(-1).detach().cpu())

    all_true_label = torch.cat(all_true_label, dim=0)
    all_pred_label = torch.cat(all_pred_label, dim=0)

    logs = utilss.update_logs(logs, iter_count)
    logs["train_pre"] = np.array([precision_score(all_true_label, all_pred_label, average='micro')])
    logs["train_rec"] = np.array([recall_score(all_true_label, all_pred_label, average='micro')])
    logs["train_f1"] = np.array([f1_score(all_true_label, all_pred_label, average='micro')])
    logs["train_f2"] = np.array([fbeta_score(all_true_label, all_pred_label, beta=2.0, average='micro')])
    utilss.show_downstream_logs("Average Training Performance:", logs)

    return logs, batch_tot_loss


def val_step(data_loader,
             ssl_model,
             downstream_classifier,
             downstream_model,
             args):

    ssl_model.eval()
    downstream_classifier.eval()
    downstream_model.eval()

    logs, last_logs = {}, None
    iter_count = 0
    all_true_label = []
    all_pred_label = []

    for step, full_data in enumerate(tqdm(data_loader, disable=args.tqdm_disable)):
        batch_data, label = full_data
        batch_data = batch_data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        batch_representation = []
        with torch.no_grad():
            for batch_idx in range(batch_data.size(0)):
                _, after_gAR, _ = ssl_model(batch_data[batch_idx], train_stage=False)
                # after_gAR.size(): time_span * channel_num * seq_size * dim_ar

                if args.direction == 'single':
                    # pooling
                    r_max = torch.max(after_gAR[:, :, :, :args.hidden_dim], dim=2)[0]
                    r_sum = torch.sum(after_gAR[:, :, :, :args.hidden_dim], dim=2)
                    r_mean = torch.mean(after_gAR[:, :, :, :args.hidden_dim], dim=2)

                    concat_representation = torch.cat((r_max, r_sum, r_mean), dim=-1)
                    after_downAR = downstream_model(concat_representation)
                    batch_representation.append(after_downAR)

                else:
                    raise Exception("Bi-direction code have not been optimized!")

            batch_representation = torch.stack(batch_representation, dim=0)

            all_losses, all_acc, true_label_batch, pred_label_batch = downstream_classifier(batch_representation, label, True)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(1)
            logs["locAcc_val"] = np.zeros(1)

        iter_count += 1
        logs["locLoss_val"] += (all_losses).detach().cpu().numpy()
        logs["locAcc_val"] += (all_acc).cpu().numpy()

        all_true_label.append(true_label_batch.view(-1).detach().cpu())
        all_pred_label.append(pred_label_batch.view(-1).detach().cpu())

    all_true_label = torch.cat(all_true_label, dim=0)
    all_pred_label = torch.cat(all_pred_label, dim=0)

    logs = utilss.update_logs(logs, iter_count)
    logs["val_pre"] = np.array([precision_score(all_true_label, all_pred_label, average='micro')])
    logs["val_rec"] = np.array([recall_score(all_true_label, all_pred_label, average='micro')])
    logs["val_f1"] = np.array([f1_score(all_true_label, all_pred_label, average='micro')])
    logs["val_f2"] = np.array([fbeta_score(all_true_label, all_pred_label, beta=2.0, average='micro')])
    utilss.show_downstream_logs("Average Validation Performance:", logs)

    return logs


def test_step(data_loader,
              ssl_model,
              downstream_classifier,
              downstream_model,
              args):

    ssl_model.eval()
    downstream_classifier.eval()
    downstream_model.eval()

    true_label = []
    pred_label = []
    soft_label = []
    for step, full_data in enumerate(data_loader):
        batch_data, label = full_data
        batch_data = batch_data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        batch_representation = []
        with torch.no_grad():
            for batch_idx in range(batch_data.size(0)):
                _, after_gAR, _ = ssl_model(batch_data[batch_idx], train_stage=False)
                # after_gAR.size(): time_span * channel_num * seq_size * dim_ar

                if args.direction == 'single':
                    # pooling
                    r_max = torch.max(after_gAR[:, :, :, :args.hidden_dim], dim=2)[0]
                    r_sum = torch.sum(after_gAR[:, :, :, :args.hidden_dim], dim=2)
                    r_mean = torch.mean(after_gAR[:, :, :, :args.hidden_dim], dim=2)

                    concat_representation = torch.cat((r_max, r_sum, r_mean), dim=-1)
                    after_downAR = downstream_model(concat_representation)
                    batch_representation.append(after_downAR)

                else:
                    raise Exception("Bi-direction code have not been optimized!")

            batch_representation = torch.stack(batch_representation, dim=0)
            all_losses, all_acc, true_label_batch, pred_label_batch, soft_label_batch = downstream_classifier(batch_representation, label, True, cal_auc=True)

        true_label.extend(true_label_batch.cpu().numpy().tolist())
        pred_label.extend(pred_label_batch.cpu().numpy().tolist())
        soft_label.extend(soft_label_batch.cpu().numpy().tolist())

    return true_label, pred_label, soft_label


class metric_index():
    def __init__(self, acc, micro_pre, micro_rec, micro_f1, micro_f_d, auc):
        self.acc = acc
        self.micro_pre = micro_pre
        self.micro_rec = micro_rec
        self.micro_f1 = micro_f1
        self.micro_f_d = micro_f_d
        self.auc = auc

def print_metrics(true_label, pred_label, soft_label):
    acc = accuracy_score(true_label, pred_label)
    pre = precision_score(true_label, pred_label, average='micro')
    rec = recall_score(true_label, pred_label, average='micro')
    f1 = f1_score(true_label, pred_label, average='micro')
    f2 = fbeta_score(true_label, pred_label, beta=2.0, average='micro')
    auc = roc_auc_score(true_label, soft_label, multi_class='ovo')
    print("-" * 40)
    print("accuracy:", acc)
    print("precision:", pre)
    print("recall:", rec)
    print("f1 score:", f1)
    print("f2 score:", f2)
    print("auc score:", auc)
    print("-" * 40)
    index = metric_index(acc, pre, rec, f1, f2, auc)
    return index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training downstream task classifier.')
    # ssl-model mode setting
    parser.add_argument('--ar_mode', type=str, default='LSTM',
                        help="The used AR model, including ['RNN', 'LSTM', 'GRU', 'TRANSFORMER']")
    parser.add_argument('--direction', type=str, default='single',
                        help="The direction for prediction task, including ['single', 'bi', 'no']")
    parser.add_argument('--graph_construct', type=str, default='sample_from_distribution',
                        help="The method for graph construction, including ['sample_from_distribution']")
    parser.add_argument('--graph_threshold', type=float, default=0.5,
                        help='The threshold to sample edges in graph construct module.')
    # hyper parameters setting
    parser.add_argument('--n_predicts', type=int, default=8,
                        help='Number of time steps in prediction task.')
    parser.add_argument('--replace_ratio', type=float, default=0.15,
                        help='The ratio for replacing timestamps in replacement task.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The hidden dimension of model.')
    parser.add_argument('--kernel_size', type=int ,nargs='+', default=[4, 4, 3],
                        help='The kernel size list for encoder (Conv1d).')
    parser.add_argument('--stride_size', type=int, nargs='+', default=[2, 2, 1],
                        help='The stride size list for encoder (Conv1d).')
    parser.add_argument('--padding_size', type=int, nargs='+', default=[0, 0, 0],
                        help='The padding size list for encoder (Conv1d).')
    # training details setting
    parser.add_argument('--gpu', action='store_false',
                        help='Whether to use gpu.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size of input data.')
    parser.add_argument('--repeat_time', type=int, default=5,
                        help='Repeat time of experiments')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The total number of epochs in the training stage.')
    parser.add_argument('--save_step', type=int, default=5,
                        help='The number of steps to save the checkpoint.')
    parser.add_argument('--early_stopping_epochs', type=int, default=10,
                        help='The number of epochs to stop training.')
    parser.add_argument('--batch_num_to_step', type=int, default=64,
                        help='The number of batches to step optimizer due to gpu-memory lack.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay of optimizer.')
    parser.add_argument('--tqdm_disable', action='store_true',
                        help='Whether disable the bar of tqdm module.')
    parser.add_argument('--ssl_stage', action='store_true',
                        help='Whether this the Self-supervised Learning stage.')
    parser.add_argument('--fine_tune', action='store_false',
                        help='Whether fine tune the ssl model.')
    parser.add_argument('--ssl_dir', type=str, default='/data/EEG_emotion_checkpoints/',
                        help='The path of SSL model to load.')
    parser.add_argument('--save_dir', type=str, default='/data/EEG_emotion_downstream_checkpoints/',
                        help='The path for saving checkpoint of downstream model.')
    parser.add_argument('--data_dir', type=str, default='/data/EEG_Emotion_Recognition/',
                        help='The path for saving EEG Emotion dataset.')
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

    print('\n', '-' * 20, 'load dataset', '-' * 20)

    dataloaders, _ = load_dataset_detection(
        data_dir=args.data_dir,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=0,
        data_type='train'
    )

    dataloaders_test, _ = load_dataset_detection(
        data_dir=args.data_dir,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=0,
        data_type='test'
    )

    index_arr = []
    random_seed_arr = [10, 100, 190, 1000, 1900]

    for repeat_idx in range(args.repeat_time):
        print(f'Repeat Time: {repeat_idx}')
        # random_seed = random.randint(0, 2 ** 31)
        random_seed = random_seed_arr[repeat_idx]
        utilss.set_seed(random_seed)
        print("Random seed:", random_seed)

        logs = {"epoch": [], "iter": [], "saveStep": args.save_step, "logging_step": 1000}

        ssl_model = MBrain(
            hidden_dim=args.hidden_dim,
            channel_num=62,
            gcn_dim=[256],
            n_predicts=args.n_predicts,
            graph_construct=args.graph_construct,
            direction=args.direction,
            replace_ratio=args.replace_ratio,
            ar_mode=args.ar_mode,
            args=args,
        )

        downstream_classifier = LinearClassifier4Emotion(
            input_dim=args.hidden_dim * 3,
            hidden_dim=[256, 128, 3],
            weighted=False
        )

        downstream_model = DownstreamCriterion(
            input_dim=downstream_classifier.input_dim,
            bi_direction=False,
        )

        if args.ssl_dir[-3:] != '.pt':
            final_epoch = 0
            for file in os.listdir(args.ssl_dir):
                if file[-3:] == '.pt' and int(file[11:][:-3]) > final_epoch:
                    final_epoch = int(file[11:][:-3])
            args.ssl_dir = os.path.join(args.ssl_dir, f"checkpoint_{final_epoch}.pt")
            print(f"\nLoading checkpoint from: {args.ssl_dir}\n")

        state_dict = torch.load(str(args.ssl_dir), 'cpu')
        ssl_model.load_state_dict(state_dict["BestModel"], strict=True)

        ssl_model.cuda()
        downstream_classifier.cuda()
        downstream_model.cuda()

        if args.fine_tune:
            print("Fine tune ssl model.")
            s_params = list(ssl_model.parameters())
            d_params = list(downstream_classifier.parameters())
            m_params = list(downstream_model.parameters())
            optimizer = torch.optim.Adam([{'params': d_params, 'lr': 5e-4}, {'params': m_params, 'lr': 5e-4},
                                          {'params': s_params, 'lr': 2e-5}],
                                         betas=(0.9, 0.999),
                                         eps=1e-08,
                                         weight_decay=args.weight_decay)
        else:
            d_params = list(downstream_classifier.parameters())
            m_params = list(downstream_model.parameters())
            optimizer = torch.optim.Adam([{'params': d_params, 'lr': 1e-3}, {'params': m_params, 'lr': 5e-4}],
                                         betas=(0.9, 0.999),
                                         eps=1e-08,
                                         weight_decay=args.weight_decay)

        # optimizer.load_state_dict(state_dict["optimizer"])

        # Checkpoint
        path_checkpoint = str(args.save_dir)
        if path_checkpoint is not None:
            if not os.path.exists(path_checkpoint):
                os.mkdir(path_checkpoint)
            else:
                shutil.rmtree(path_checkpoint)
                os.mkdir(path_checkpoint)

            path_checkpoint = os.path.join(path_checkpoint, "checkpoint")

        # print("Loading SSL model from:", str(args.ssl_dir))
        print(f"Running {args.epochs} epochs")
        n_epoch = args.epochs
        start_epoch = 0
        loss_increase_epoch = 0
        last_loss = 0
        best_val_loss = np.inf
        best_val_metric = 0
        start_time = time.time()
        training_ending_signal = False

        train_loader = dataloaders['train']
        val_loader = dataloaders['dev']
        print("Training Dataset: %d batches, Validation Dataset: %d batches, Batch Size: %d" %
              (len(train_loader), len(val_loader), args.batch_size))

        for epoch in range(start_epoch, n_epoch):
            print(f"\nStarting epoch {epoch + 1}")

            loc_logs_train, current_loss = \
                train_step(train_loader, ssl_model, downstream_classifier,
                           downstream_model, optimizer, args)

            loc_logs_val = \
                val_step(val_loader, ssl_model, downstream_classifier,
                         downstream_model, args)

            print(f'Ran {epoch + 1} epochs '
                  f'in {time.time() - start_time:.2f} seconds')

            loss_change = np.fabs(current_loss - last_loss)
            last_loss = current_loss
            # current_val_loss = float(loc_logs_val["locLoss_val"].mean())
            current_val_metric = float(loc_logs_val["val_f1"].mean())
            loss_increase_epoch += 1

            if current_val_metric > best_val_metric:
                best_model_state = deepcopy(ssl_model.state_dict())
                best_criterion_state = deepcopy(downstream_classifier.state_dict())
                best_downstreamModel_state = deepcopy(downstream_model.state_dict())
                best_val_metric = current_val_metric
                loss_increase_epoch = 0
                print('Save current model!')

            for key, value in dict(loc_logs_train, **loc_logs_val).items():
                if key not in logs:
                    logs[key] = [None for _ in range(epoch)]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                logs[key].append(value)

            logs["epoch"].append(epoch)

            if loss_change < 0.001 or loss_increase_epoch >= args.early_stopping_epochs:
                training_ending_signal = True

            if path_checkpoint is not None \
                    and ((epoch+1) % logs["saveStep"] == 0 or epoch == n_epoch-1 or training_ending_signal):
                ssl_state = ssl_model.state_dict()
                criterion_state = downstream_classifier.state_dict()
                downstreamModel_state = downstream_model.state_dict()
                optimizer_state = optimizer.state_dict()

                utilss.save_checkpoint_newDownstream(ssl_state, criterion_state, optimizer_state,
                                                    downstreamModel_state, best_downstreamModel_state,
                                                    best_model_state, best_criterion_state, best_val_metric,
                                                    f"{path_checkpoint}_{epoch + 1}.pt")
                utilss.save_logs(logs, path_checkpoint + "_logs.json")
                print('Logs updated!')

            if training_ending_signal:
                break

        print("After %d epochs training, ending!"%(epoch + 1))

        ##############################################################################
        ################################# test stage #################################
        ##############################################################################
        test_loader = dataloaders_test['test']

        final_epoch = 0
        for file in os.listdir(args.save_dir):
            if file[-3:] == '.pt' and int(file[11:][:-3]) > final_epoch:
                final_epoch = int(file[11:][:-3])
        # Since early stop is 10, we have to decrease right margin
        if final_epoch > args.early_stopping_epochs:
            final_epoch -= args.early_stopping_epochs
            while final_epoch % 5:
                final_epoch += 1

        checkpoint = [5, final_epoch]
        print(f"\nCheckpoints range: [{checkpoint[0]}, {checkpoint[1]}]\n")

        checkpoint_dir = os.path.join(args.save_dir, f"checkpoint_{final_epoch}.pt")
        print("\nLoading checkpoint from:", checkpoint_dir)

        state_dict = torch.load(checkpoint_dir, 'cpu')
        ssl_model.load_state_dict(state_dict["BestModel"], strict=True)
        downstream_classifier.load_state_dict(state_dict["BestCriterion"], strict=True)
        downstream_model.load_state_dict(state_dict["BestDownstreamModel"], strict=True)

        ssl_model.cuda()
        downstream_classifier.cuda()
        downstream_model.cuda()

        channel_true_label, channel_pred_label, soft_label = test_step(test_loader, ssl_model, downstream_classifier, downstream_model, args)

        index = print_metrics(channel_true_label, channel_pred_label, soft_label)
        index_arr.append(index)

        print("-" * 40)
        print("accuracy:", index.acc)
        print("precision:", index.micro_pre)
        print("recall:", index.micro_rec)
        print("f1 score:", index.micro_f1)
        print("f2 score:", index.micro_f_d)
        print("auc score:", index.auc)
        print("-" * 40)

    acc_arr = [index.acc for index in index_arr if index]
    pre_arr = [index.micro_pre for index in index_arr if index]
    rec_arr = [index.micro_rec for index in index_arr if index]
    f1_arr = [index.micro_f1 for index in index_arr if index]
    f2_arr = [index.micro_f_d for index in index_arr if index]
    auc_arr = [index.auc for index in index_arr if index]
    print('-' * 48)
    print(f"Micro performance:\n"
          f"Accuracy:  mean:{np.mean(acc_arr)} std:{np.std(acc_arr)}\n"
          f"Precision: mean:{np.mean(pre_arr)} std:{np.std(pre_arr)}\n"
          f"Recall:    mean:{np.mean(rec_arr)} std:{np.std(rec_arr)}\n"
          f"F1-Score:  mean:{np.mean(f1_arr)} std:{np.std(f1_arr)}\n"
          f"F2-Score:  mean:{np.mean(f2_arr)} std:{np.std(f2_arr)}\n"
          f"AUROC:     mean:{np.mean(auc_arr)} std:{np.std(auc_arr)}")
    print('-' * 48)
