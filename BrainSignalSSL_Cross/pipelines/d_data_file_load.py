import os
import pandas as pd
import numpy as np
import pickle
from scipy import signal
from mne.io import read_raw_edf


class RawDataLoader:
    def __init__(
            self,
            root_path,
            file,
            normalize=True,
            all_channel=False,
            down_sampling=True,
            filter=False,
    ):
        raw_channel_name_path = os.path.join(root_path, file, file + '_channelname.csv')
        minus_channel_name_path = os.path.join(root_path, 'minus_label_dict.pkl')
        all_minus_channel_name_path = os.path.join(root_path, 'all_minus_label_dict.pkl')
        sample_rate_path = os.path.join(root_path, 'sample_rate_dict.pkl')
        del_channel_path = os.path.join(root_path, 'del_channels.pkl')

        self.raw_data_path = None
        self.raw_data = None
        # Correct the order and remove the illegal channels
        self.raw_name_dict = pd.read_csv(raw_channel_name_path)
        self.raw_name_dict['channelName'] = self.raw_name_dict.channelName.apply(lambda x: x.replace(' ', ''))
        self.raw_name_dict.set_index('channelName', inplace=True)
        # Only include the minus channels not in the del_channel_list
        with open(minus_channel_name_path, 'rb') as f:
            self.minus_name_dict = pickle.load(f)
        # Include all the minus channels of one patient
        with open(all_minus_channel_name_path, 'rb') as f:
            self.all_minus_name_dict = pickle.load(f)
        with open(sample_rate_path, 'rb') as f:
            sample_rate_dict = pickle.load(f)
            self.sample_rate = sample_rate_dict[file]
            if down_sampling:
                self.base_sample_rate = 250
                if self.sample_rate % self.base_sample_rate != 0:
                    raise ValueError('The base sample rate cannot divide the sample rate of', os.path.join(root_path, file))
            else:
                self.base_sample_rate = sample_rate_dict['base']
        # Not use now
        with open(del_channel_path, 'rb') as f:
            self.del_channel_list = pickle.load(f)
        self.root_path = root_path
        self.file = file
        self.normalize = normalize
        self.all_channel = all_channel
        self.down_sampling = down_sampling
        self.filter = filter

    def __load_raw_data__(self, minus_data_flag=False):
        if minus_data_flag:
            if self.raw_data is None:
                raw_data_path = os.path.join(self.root_path, self.file, self.file + '.edf')
                self.raw_data_path = raw_data_path
                self.raw_data = read_raw_edf(raw_data_path, preload=False)[:][0]

            raw_data = self.raw_data
            if self.filter:
                raw_data = self.__filter_data__(raw_data)

            minus_name_dic = self.minus_name_dict if not self.all_channel else self.all_minus_name_dict
            raw_minus_data = np.empty([len(minus_name_dic), raw_data.shape[1]], dtype=np.float)
            for key, value in minus_name_dic.items():
                ch_name = key.split('-')
                count = value[0]
                raw_minus_data[count] = raw_data[self.raw_name_dict.loc[ch_name[1], 'index']] - raw_data[
                    self.raw_name_dict.loc[ch_name[0], 'index']]
            del raw_data
            return raw_minus_data
        else:
            if self.raw_data is None:
                raw_data_path = os.path.join(self.root_path, self.file, self.file + '.edf')
                self.raw_data_path = raw_data_path
                self.raw_data = read_raw_edf(raw_data_path, preload=False)[:][0]
            raw_data = self.raw_data
            if self.filter:
                raw_data = self.__filter_data__(raw_data)
            return raw_data

    def __filter_data__(self, data):
        # bandpass filter
        print("Filter data!!!")
        cut_freq = np.array([0.5, 70])
        nyq = self.sample_rate / 2
        sos = signal.iirfilter(2, cut_freq / nyq, btype='bandpass', ftype='butter', output='sos')
        data = signal.sosfiltfilt(sos, data, axis=-1)
        # bandstop filter
        cut_freq = np.array([49, 51])
        sos = signal.iirfilter(2, cut_freq / nyq, btype='bandstop', ftype='butter', output='sos')
        data = signal.sosfiltfilt(sos, data, axis=-1)

        return data

    def single_data_loader(self, raw_data=None, re_filter=False):
        if raw_data is None:
            raw_data = self.__load_raw_data__(minus_data_flag=False)
        elif re_filter:
            raw_data = self.__filter_data__(raw_data)
        # TODO: We have to remove the bad channels in del_channel_list.
        #  However, the list includes minus channels and the single channel data cannot use now.
        if self.normalize:
            mu = raw_data.mean(axis=-1).reshape(-1, 1)
            sigma = raw_data.std(axis=-1).reshape(-1, 1)
            return (raw_data - mu) / sigma
        else:
            return raw_data

    def minus_data_loader(self, raw_minus_data=None, raw_data=None, re_filter=False):
        if raw_minus_data is None:
            if raw_data is not None:
                self.raw_data = raw_data
            raw_minus_data = self.__load_raw_data__(minus_data_flag=True)
        elif re_filter:
            raw_minus_data = self.__filter_data__(raw_minus_data)
        if self.normalize:
            mu = raw_minus_data.mean(axis=-1).reshape(-1, 1)
            sigma = raw_minus_data.std(axis=-1).reshape(-1, 1)
            return (raw_minus_data - mu) / sigma
        else:
            return raw_minus_data


class RawLabelLoader:
    def __init__(
            self,
            root_path,
            file,
            all_channel=False,
            down_sampling=True,
    ):
        all_label_path = os.path.join(root_path, file, 'data_label.pkl')
        minus_label_path = os.path.join(root_path, file, 'data_minus_label.pkl')
        if down_sampling:
            self.base_sample_rate = 250
        minus_channel_name_path = os.path.join(root_path, 'minus_label_dict.pkl')
        sample_rate_path = os.path.join(root_path, 'sample_rate_dict.pkl')
        self.all_label_path = all_label_path
        self.minus_label_path = minus_label_path
        self.all_label = None
        self.minus_label = None
        with open(minus_channel_name_path, 'rb') as f:
            self.minus_name_dic = pickle.load(f)
        with open(sample_rate_path, 'rb') as f:
            sample_rate_dict = pickle.load(f)
            self.sample_rate = sample_rate_dict[file]
            if down_sampling:
                if self.sample_rate % self.base_sample_rate != 0:
                    raise ValueError('The base sample rate cannot divide the sample rate of',
                                     os.path.join(root_path, file))
            else:
                self.base_sample_rate = sample_rate_dict['base']
        self.all_channel = all_channel
        self.down_sampling = down_sampling

    @staticmethod
    def __load_sparse_label__(sparse_label_path):
        with open(sparse_label_path, 'rb') as f:
            sparse_label = pickle.load(f)
            f.close()
        label = np.zeros((len(sparse_label[-1]), sparse_label[0]), dtype=np.int64)

        # Customized Sparse Type (Slice save)
        # [time_length, [triples] * channel_num]
        # triple: [start_index, end_index, value]
        for ch_index in range(label.shape[0]):
            ch_triples = sparse_label[-1][ch_index]
            for start_index, end_index, value in ch_triples:
                label[ch_index][start_index:end_index] = value
        return label

    def all_label_loader(self):
        if self.all_label is None:
            self.all_label = self.__load_sparse_label__(self.all_label_path).squeeze()
        return self.all_label

    def minus_label_loader(self):
        if self.minus_label is None:
            self.minus_label = self.__load_sparse_label__(self.minus_label_path)
        if self.all_channel:
            return self.minus_label
        keep_channel_list = []
        for _, value in self.minus_name_dic.items():
            keep_channel_list.append(value[1])
        return self.minus_label[keep_channel_list]
