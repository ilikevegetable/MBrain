import torch
import argparse
from tqdm import tqdm
import sys
import os
import numpy as np
import random
import utils.misc as utilss

sys.path.append('criterion')
sys.path.append('model')
sys.path.append('pipelines')

from model.ssl_model import MBrain
from criterion.downstream_task_criterion import LinearClassifier4test, NewDownstreamCriterion
from pipelines.g_dataset_dataloader import load_dataset


def test_step(data_loader,
              ssl_model,
              downstream_classifier,
              downstream_model,
              args):

    ssl_model.eval()
    downstream_classifier.eval()
    downstream_model.eval()

    channel_true_label = []
    channel_pred_label = []
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
            true_label_batch, pred_label_batch = downstream_classifier(batch_representation, label)

        batch_size, time_span, channel_num, _ = batch_data.size()
        true_label_batch = true_label_batch.view(batch_size, time_span, channel_num)
        pred_label_batch = pred_label_batch.view(batch_size, time_span, channel_num)
        true_label_batch = true_label_batch.cpu().numpy().tolist()
        pred_label_batch = pred_label_batch.cpu().numpy().tolist()
        for big_num in range(len(true_label_batch)):
            for seg_num in range(len(true_label_batch[0])):
                if len(channel_true_label) == 0:
                    channel_true_label = [[true_label_batch[big_num][seg_num][ch]] for ch in
                                          true_label_batch[big_num][seg_num]]
                    channel_pred_label = [[pred_label_batch[big_num][seg_num][ch]] for ch in
                                          pred_label_batch[big_num][seg_num]]
                else:
                    for channel_num in range(len(channel_true_label)):
                        channel_true_label[channel_num].append(true_label_batch[big_num][seg_num][channel_num])
                        channel_pred_label[channel_num].append(pred_label_batch[big_num][seg_num][channel_num])

    return channel_true_label, channel_pred_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training downstream task classifier.')
    # ssl-model mode setting
    parser.add_argument('--ar_mode', type=str, default='LSTM',
                        help="The used AR model, including ['RNN', 'LSTM', 'GRU', 'TRANSFORMER']")
    parser.add_argument('--direction', type=str, default='single',
                        help="The direction for prediction task, including ['single', 'bi', 'no']")
    parser.add_argument('--graph_construct', type=str, default='sample_from_distribution',
                        choices=['sample_from_distribution', 'noGraph'],
                        help="The method for graph construction, including ['cos', 'mi', 'gumbel', 'cos_threshold']")
    parser.add_argument('--graph_threshold', type=float, default=0.5,
                        help='The threshold to sample edges in graph construct module.')
    # hyper parameters setting
    parser.add_argument('--n_predicts', type=int, default=8,
                        help='Number of time steps in prediction task.')
    parser.add_argument('--replace_ratio', type=float, default=0.15,
                        help='The ratio for replacing timestamps in replacement task.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The hidden dimension of model.')
    parser.add_argument('--kernel_size', type=int ,nargs='+', default=[4, 4, 4],
                        help='The kernel size list for encoder (Conv1d).')
    parser.add_argument('--stride_size', type=int, nargs='+', default=[2, 2, 1],
                        help='The stride size list for encoder (Convls1d).')
    parser.add_argument('--padding_size', type=int, nargs='+', default=[0, 0, 0],
                        help='The padding size list for encoder (Conv1d).')
    # testing details setting
    # parser.add_argument('--cuda_idx', type=int, default=2,
    #                     help='The index of cuda to be used.')
    parser.add_argument('--ssl_stage', action='store_true',
                        help='Whether this the Self-supervised Learning stage.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size of input data.')
    parser.add_argument('--save_dir', type=str, default='/data/caidonghong/nips22/rebuttal/downstream/MBrain_02GJX_l1_03_l2_04/',
                        help='The path for saving checkpoint.')
    parser.add_argument('--checkpoint', type=int, nargs=2, default=[-1, -1],
                        help='The start and end checkpoint for testing.')
    parser.add_argument('--load_best_checkpoint', action='store_false',
                        help='Whether to load best checkpoint in testing.')
    # load dataset setting
    parser.add_argument('--database_save_dir', type=str, default='/data/caidonghong/new_database2_no_filter/02GJX/',
                        help='The path for database_save_dir while loading database.'
                             'Patients: [01TGX, 02GJX, 03ZXY, 05ZLH, 06ZYJ]'
                             '01:125; 02:52; 03:120; 05:116; 06:101')
    parser.add_argument('--data_save_dir', type=str, default='/data/eeggroup/new_data2/02GJX/',
                        help='The path for data_save_dir while loading database.')
    parser.add_argument('--data_normalize', action='store_false',
                        help='Whether to normalize the data.')
    parser.add_argument('--data_multi_level', action='store_true',
                        help='Whether to get multi level labels of data.')
    parser.add_argument('--infer', action='store_true',
                        help='Whether to infer whole testing file.')
    args = parser.parse_args()

    print('-' * 20, 'print parameters', '-' * 20)
    for arg in vars(args):
        print(format(arg, '<25'), format(str(getattr(args, arg)), '<'))  # str, arg_type

    print('\n', '-' * 20, 'load dataset', '-' * 20)
    # random_seed = random.randint(0, 2 ** 31)
    random_seed = 10
    utilss.set_seed(random_seed)
    print("Random seed:", random_seed)

    if args.infer:
        infer_file_dict = {
            "01TGX": "FA0014KI",
            "02GJX": "FA00127W",
            "03ZXY": "FA00145X",
            "05ZLH": "FA0010AP",
            "06ZYJ": "FA0011MB",
        }
        patient = args.data_save_dir.split('/')[-2] if args.data_save_dir.split('/')[-1] == '' else args.data_save_dir.split('/')[-1]
        args.data_save_dir = os.path.join(args.data_save_dir, infer_file_dict[patient])
        data_type = 'infer'
    else:
        data_type = 'test'

    if args.checkpoint[0] < 5:
        final_epoch = 0
        for file in os.listdir(args.save_dir):
            if file[-3:] == '.pt' and int(file[11:][:-3]) > final_epoch:
                final_epoch = int(file[11:][:-3])
        # Since early stop is 10, we have to decrease right margin
        if args.load_best_checkpoint:
            final_epoch -= 10
            while final_epoch%5:
                final_epoch += 1

        args.checkpoint = [5, final_epoch]
        print(f"\nCheckpoints range: [{args.checkpoint[0]}, {args.checkpoint[1]}]\n")


    test_dataset = load_dataset(
        database_save_dir=args.database_save_dir,
        data_save_dir=args.data_save_dir,
        window_time=1,
        slide_time=1,
        data_type=data_type,
        test_ratio=50,
        channel_list=None,
        normalize=args.data_normalize,
        multi_level=args.data_multi_level,
        shared_encoder=True,
        n_process_loader=50,
    )

    ssl_model = MBrain(
        hidden_dim=args.hidden_dim,
        gcn_dim=[256],
        n_predicts=args.n_predicts,
        graph_construct=args.graph_construct,
        direction=args.direction,
        replace_ratio=args.replace_ratio,
        ar_mode=args.ar_mode,
        args=args,
    )


    downstream_classifier = LinearClassifier4test(
        input_dim=args.hidden_dim * 3,
        hidden_dim=[256, 128, 2],
        weighted=False
    )

    downstream_model = NewDownstreamCriterion(
        input_dim=downstream_classifier.input_dim,
        bi_direction=False,
    )

    test_loader = test_dataset.get_data_loader(args.batch_size, num_workers=0)
    # torch.cuda.set_device(args.cuda_idx)

    best_index = None
    best_f2 = 0
    best_file_path = None

    for i in range(args.checkpoint[0], args.checkpoint[1] + 5, 5):
        i = i if i < args.checkpoint[1] else args.checkpoint[1]
        checkpoint_dir = os.path.join(args.save_dir, f"checkpoint_{i}.pt")
        print("\nLoading checkpoint from:", checkpoint_dir)

        state_dict = torch.load(checkpoint_dir, 'cpu')
        if args.load_best_checkpoint:
            ssl_model.load_state_dict(state_dict["BestModel"], strict=True)
            downstream_classifier.load_state_dict(state_dict["BestCriterion"], strict=True)
            downstream_model.load_state_dict(state_dict["BestDownstreamModel"], strict=True)
        else:
            ssl_model.load_state_dict(state_dict["sslModel"], strict=True)
            downstream_classifier.load_state_dict(state_dict["downstreamCriterion"], strict=True)
            downstream_model.load_state_dict(state_dict["downstreamModel"], strict=True)

        ssl_model.cuda()
        downstream_classifier.cuda()
        downstream_model.cuda()

        channel_true_label, channel_pred_label = test_step(test_loader, ssl_model, downstream_classifier,
                                                           downstream_model, args)

        if args.infer:
            channel_true_label = np.array(channel_true_label)
            channel_true_label = np.where(channel_true_label > 1, 0, channel_true_label)
            channel_true_label = np.where(channel_true_label < 0, 2, channel_true_label)

        index = test_dataset.data_handler.model_evaluation(channel_true_label, channel_pred_label, None, 'label')
        if index.micro_f_d > best_f2:
            best_index = index
            best_f2 = index.micro_f_d
            best_file_path = checkpoint_dir
        print(index)

    print(f"Best checkpoint is from: {best_file_path}")
    print(best_index)

