import os
import numpy as np
import pickle
import pandas as pd
import copy

from .fb_evaluation_api import Evaluator, EvaluationIndex


class DataPack:
    def __init__(self):
        # basic content
        self.data = None
        self.true_all_label = None
        self.true_minus_label = None
        self.loc = None
        self.normalize = None
        self.ch_loc = None

        # used to locate the data
        self.file_name = None
        self.data_type = None


class DataHandler:
    def __init__(
            self,
            database_save_dir='/data/eeggroup/new_database2/02GJX/',   # The path of database file holder
            data_save_dir='/data/eeggroup/new_data2/02GJX/',           # The path of original data file holder
            window_time=2,          # The unit is second
            slide_time=1,           # The unit is second
            down_sampling=True,
    ):
        self.database_save_dir = database_save_dir
        self.data_save_dir = data_save_dir

        sample_rate_path = os.path.join(self.data_save_dir, 'sample_rate_dict.pkl')
        with open(sample_rate_path, 'rb') as f:
            sample_rate_dict = pickle.load(f)
            base_sample_rate = sample_rate_dict['base']
            if down_sampling:
                if base_sample_rate % 250 == 0:
                    self.sample_rate = 250
                else:
                    raise ValueError('The base sample rate is not a multiple of 250.')
            else:
                self.sample_rate = base_sample_rate
            f.close()

        minus_channel_name_path = os.path.join(self.data_save_dir, 'minus_label_dict.pkl')
        with open(minus_channel_name_path, 'rb') as f:
            minus_name_dic = pickle.load(f)
            self.brain_dict = minus_name_dic
            self.channel_num = len(minus_name_dic)
            print('Channel number of the patient is:', self.channel_num)

        self.slide_len = int(slide_time * self.sample_rate)
        self.window_len = int(window_time * self.sample_rate)
        self.seg_num = 1
        self.evaluator = Evaluator()

    @staticmethod
    def __get_data_from_ch_list__(data, ch_list):
        if ch_list is None:
            new_data = data
        else:
            new_data = [ch[ch_list] for ch in data]
        return new_data

    def __get_database__(
            self,
            data_type,
            ch_list=None,
            normalize=True,
    ):
        assert data_type in ['train', 'valid', 'test_ratio500', 'test_ratio50', 'test_ratio5', 'clean', 'valid_clean']
        data_pack = DataPack()
        clean_flag = False
        # if data_type == 'clean' or data_type == 'valid_clean':
        #     ch_list = None
        #     clean_flag = True

        if normalize:
            data = self.__get_data_from_ch_list__(np.load(
                os.path.join(self.database_save_dir, data_type + '_normal_minus_data.npz'))['x'], ch_list)
        else:
            data = self.__get_data_from_ch_list__(np.load(
                os.path.join(self.database_save_dir, data_type + '_minus_data.npz'))['x'], ch_list)
        data = [np.float64(seq) for seq in data]

        all_label = np.load(os.path.join(self.database_save_dir, data_type + '_all_label.npz'))['x']
        minus_label = self.__get_data_from_ch_list__(
            np.load(os.path.join(self.database_save_dir, data_type + '_minus_label.npz'))['x'], ch_list)
        with open(os.path.join(self.database_save_dir, data_type + '_loc.pkl'), 'rb') as f:
            all_loc_data = pickle.load(f)
            file_name = [loc[0] for loc in all_loc_data]
            if clean_flag:
                loc_data = [[loc[1], np.arange(*loc[2])] for loc in all_loc_data]
            else:
                loc_data = [np.arange(*loc[1]) for loc in all_loc_data]
            f.close()

        data_pack.true_all_label = np.array(all_label)
        data_pack.true_minus_label = np.array(minus_label)
        if clean_flag:
            ch_loc = [loc[0] for loc in loc_data]
            loc_data = [loc[1] for loc in loc_data]
            data_pack.ch_loc = np.array(ch_loc)
        data_pack.loc = np.array(loc_data)

        data_pack.data = np.array(data)
        data_pack.file_name = np.array(file_name)
        data_pack.data_type = data_type
        data_pack.normalize = normalize
        return data_pack

    def get_train_data(self, ch_list=None, normalize=True):
        return self.__get_database__(
            'train',
            ch_list=ch_list,
            normalize=normalize,
        )

    def get_valid_data(self, ch_list=None, normalize=True):
        return self.__get_database__(
            'valid',
            ch_list=ch_list,
            normalize=normalize,
        )

    def get_test_data(self, ratio, ch_list=None, normalize=True):
        assert ratio in [500, 50, 5, '500', '50', '5']
        return self.__get_database__(
            'test_ratio' + str(ratio),
            ch_list=ch_list,
            normalize=normalize,
        )

    def get_clean_data(self, ch_list=None, normalize=True):
        return self.__get_database__(
            'clean',
            ch_list=ch_list,
            normalize=normalize,
        )

    def get_valid_clean_data(self, ch_list=None, normalize=True):
        return self.__get_database__(
            'valid_clean',
            ch_list=ch_list,
            normalize=normalize,
        )

    def get_segment_data(self, data_pack, silence=False):
        data_list, all_label_list, minus_label_list, loc_list, file_list = \
            data_pack.data, data_pack.true_all_label, data_pack.true_minus_label, data_pack.loc, data_pack.file_name
        seg_data = []
        seg_all_label = []
        seg_minus_label = []
        seg_loc = []
        seg_file = []
        new_data_pack = DataPack()
        data_list = np.array(data_list)
        for i in range(data_list.shape[0]):
            data_block = data_list[i]
            start = 0
            end = self.window_len
            while end <= data_block.shape[1]:
                seg_data.append(data_block[:, start:end])
                if loc_list is not None:
                    seg_loc.append(loc_list[i][start:end])
                if file_list is not None:
                    seg_file.append(file_list[i])
                if all_label_list is not None:
                    seg_all_label.append(all_label_list[i][start:end].max())
                if minus_label_list is not None:
                    if data_pack.data_type == 'infer':
                        seg_minus_label.append(minus_label_list[i][:, start:end])
                    else:
                        seg_minus_label.append(minus_label_list[i][:, start:end].max(axis=-1))

                start += self.slide_len
                end += self.slide_len
        # Compute the number of small segments divided from every big segment
        self.seg_num = (data_list.shape[-1] - self.window_len) // self.slide_len + 1

        if not silence:
            print('Total ' + data_pack.data_type + ' Segment Number: ', len(seg_data))

        new_data_pack.data = np.array(seg_data)
        if loc_list is not None:
            new_data_pack.loc = np.array(seg_loc)
        if file_list is not None:
            new_data_pack.file_name = np.array(seg_file)
        if all_label_list is not None:
            new_data_pack.true_all_label = np.array(seg_all_label)
        if minus_label_list is not None:
            new_data_pack.true_minus_label = np.array(seg_minus_label)
        new_data_pack.data_type = data_pack.data_type
        if data_pack.normalize is not None:
            new_data_pack.normalize = data_pack.normalize
        return new_data_pack

    def get_dataframe_sktime_segment_data(self, data_pack):
        new_data_pack = self.get_segment_data(data_pack)
        seg_data = new_data_pack.data

        df = pd.DataFrame()
        for dim in range(seg_data.shape[1]):
            case_list = []
            for data_block in seg_data:
                case_list.append(pd.Series(data_block[dim]))
            df['dim_' + str(dim)] = case_list

        return df, copy.deepcopy(new_data_pack.true_all_label), new_data_pack

    def segment_label_evaluation(self, y_true, y_pred, positive_pred=None, valid_flag=True):
        return self.evaluator.class_evaluation(y_true, y_pred, positive_pred, valid_flag)

    def segment_prob_evaluation(self, y_true, y_prob, valid_flag=True):
        return self.evaluator.prob_evaluation(y_true, y_prob, valid_flag)

    def model_evaluation(self, true_label, pred_prob, positive_pred_prob=None, mode='label'):
        index = EvaluationIndex()
        true_label = np.array(true_label)
        pred_prob = np.array(pred_prob)
        if positive_pred_prob is not None:
            positive_pred_prob = np.array(positive_pred_prob)
        if true_label.shape[0] > 1:
            index.macro_tp, index.macro_fp, index.macro_fn, index.macro_tn, index.macro_pre, index.macro_rec, \
                index.macro_f_h, index.macro_f1, index.macro_f_d = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            if mode == 'prob' or positive_pred_prob is not None:
                index.macro_auc = 0.

        valid_channel_list = []
        # Only the channels with at least one `1` sample are meaningful to compute the macro index
        for i in range(len(true_label)):
            # The '2' labels are the invalid labels
            valid_index = np.where(true_label[i] != 2)[0]
            valid_flag = (np.sum(true_label[i][valid_index]) > 0)
            if valid_flag:
                valid_channel_list.append(i)
                if index.macro_pre is None:
                    continue
                if mode == 'label':
                    if positive_pred_prob is None:
                        macro_index = self.segment_label_evaluation(true_label[i][valid_index],
                                                                    pred_prob[i][valid_index], None, valid_flag)
                    else:
                        macro_index = self.segment_label_evaluation(true_label[i][valid_index],
                                                                    pred_prob[i][valid_index],
                                                                    positive_pred_prob[i][valid_index], valid_flag)
                elif mode == 'prob':
                    macro_index = self.segment_prob_evaluation(true_label[i][valid_index], pred_prob[i][valid_index],
                                                               valid_flag)
                else:
                    raise ValueError(f'The mode {mode} is not in the valid list [label, prob].')

                index.macro_tp += macro_index.micro_tp
                index.macro_fp += macro_index.micro_fp
                index.macro_fn += macro_index.micro_fn
                index.macro_tn += macro_index.micro_tn
                index.macro_pre += macro_index.micro_pre
                index.macro_rec += macro_index.micro_rec
                index.macro_f_h += macro_index.micro_f_h
                index.macro_f1 += macro_index.micro_f1
                index.macro_f_d += macro_index.micro_f_d
                if index.macro_auc is not None:
                    index.macro_auc += macro_index.micro_auc

        # The micro mode should consider all the channels with or without '1' labels.
        # The '2' labels are the invalid labels
        valid_index = np.where(true_label.reshape(-1) != 2)[0]
        if mode == 'label':
            if positive_pred_prob is None:
                micro_index = self.segment_label_evaluation(true_label.reshape(-1)[valid_index],
                                                            pred_prob.reshape(-1)[valid_index], None)
            else:
                micro_index = self.segment_label_evaluation(true_label.reshape(-1)[valid_index],
                                                            pred_prob.reshape(-1)[valid_index],
                                                            positive_pred_prob.reshape(-1)[valid_index])
        elif mode == 'prob':
            micro_index = self.segment_prob_evaluation(true_label.reshape(-1)[valid_index],
                                                       pred_prob.reshape(-1)[valid_index])
        else:
            raise ValueError(f'The mode {mode} is not in the valid list [label, prob].')

        if index.macro_pre is not None:
            valid_count = len(valid_channel_list)
            index.macro_tp /= valid_count
            index.macro_fp /= valid_count
            index.macro_fn /= valid_count
            index.macro_tn /= valid_count
            index.macro_pre /= valid_count
            index.macro_rec /= valid_count
            index.macro_f_h /= valid_count
            index.macro_f1 /= valid_count
            index.macro_f_d /= valid_count
            if index.macro_auc is not None:
                index.macro_auc /= valid_count

        index.acc = micro_index.acc
        index.micro_tp = micro_index.micro_tp
        index.micro_fp = micro_index.micro_fp
        index.micro_fn = micro_index.micro_fn
        index.micro_tn = micro_index.micro_tn
        index.micro_pre = micro_index.micro_pre
        index.micro_rec = micro_index.micro_rec
        index.micro_f_h = micro_index.micro_f_h
        index.micro_f1 = micro_index.micro_f1
        index.micro_f_d = micro_index.micro_f_d
        if micro_index.micro_auc is not None:
            index.micro_auc = micro_index.micro_auc

        return index
