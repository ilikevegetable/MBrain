import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.multiprocessing import Pool

from .fa_database_api import DataHandler, DataPack
from .d_data_file_load import RawDataLoader, RawLabelLoader
from .e_database_generate import uniform_sample_data


def load_dataset(
        database_save_dir,
        data_save_dir,
        window_time,
        slide_time,
        data_type='ssl',
        test_ratio=50,
        channel_list=None,
        normalize=True,
        multi_level=True,
        shared_encoder=True,
        n_process_loader=50,
        cal_timeShift_label=None,
        patient_list=None,
        valid_patient=None,
):
    print("Loading the", data_type, "dataset...")
    infer_file = None
    if data_type == 'infer':
        # Input data_save_dir should be: xxx/patient_name/infer_file_name
        data_save_dir, infer_file = os.path.split(data_save_dir)

    if data_type == 'mean_matrix':
        data_handler = DataHandler(
            database_save_dir=database_save_dir,
            data_save_dir=data_save_dir,
            window_time=window_time,
            slide_time=slide_time,
        )
        data_handler.multi_level = multi_level
        data_handler.shared_encoder = shared_encoder

        data_pack = data_handler.get_clean_data(ch_list=channel_list, normalize=normalize)
        seg_data_pack = data_handler.get_segment_data(data_pack)

        main_data = seg_data_pack.data
        main_label = seg_data_pack.true_minus_label
        main_ch_loc = seg_data_pack.ch_loc
        main_dataset = EEGBatchData(
            data_handler,
            main_data,
            main_label,
            main_ch_loc,
            data_type,
            n_process_loader,
            cal_timeShift_label,
        )
        return main_dataset

    elif data_type == 'test' or data_type == 'infer':
        data_handler = DataHandler(
            database_save_dir=database_save_dir,
            data_save_dir=data_save_dir,
            window_time=window_time,
            slide_time=slide_time,
        )
        data_handler.multi_level = multi_level
        data_handler.shared_encoder = shared_encoder

        assert data_type in ['ssl', 'train', 'test', 'infer']
        if data_type == 'ssl':
            data_pack = data_handler.get_clean_data(ch_list=channel_list, normalize=normalize)
        elif data_type == 'train':
            data_pack = data_handler.get_train_data(ch_list=channel_list, normalize=normalize)
        elif data_type == 'test':
            data_pack = data_handler.get_test_data(ratio=test_ratio, ch_list=channel_list, normalize=normalize)
        else:
            raw_data_loader = RawDataLoader(
                root_path=data_save_dir,
                file=infer_file,
                normalize=normalize,
            )
            raw_label_loader = RawLabelLoader(
                root_path=data_save_dir,
                file=infer_file,
            )
            minus_data = raw_data_loader.minus_data_loader()
            minus_label = raw_label_loader.minus_label_loader()
            data_handler.down_sampling = raw_data_loader.sample_rate // raw_data_loader.base_sample_rate
            data_handler.origin_length = minus_data.shape[-1]
            # Force all the files in one patient to be the same sample rate
            if raw_data_loader.sample_rate > raw_data_loader.base_sample_rate:
                _, minus_data, _, _, minus_label, _ = \
                    uniform_sample_data(raw_data_loader.sample_rate // raw_data_loader.base_sample_rate,
                                        minus_data.shape[-1], minus_data=minus_data, minus_label=minus_label)
            # Transform other positive labels to zero labels
            minus_label = np.where(minus_label > 1, 0, minus_label)
            # Transform negative labels to more than one labels
            minus_label = np.where(minus_label < 0, 2, minus_label)

            print('This patient file lasts: ', minus_data.shape[-1] / 3600 / raw_data_loader.base_sample_rate, 'hours.')

            # data_handler.window_len = int(20 * data_handler.sample_rate)
            # data_handler.slide_len = int(19 * data_handler.sample_rate)
            data_handler.window_len = int(10 * data_handler.sample_rate)
            data_handler.slide_len = int(10 * data_handler.sample_rate)
            data_pack = DataPack()
            data_pack.data = [minus_data]
            data_pack.true_minus_label = [minus_label]
            data_pack.data_type = 'infer'
            data_pack = data_handler.get_segment_data(data_pack)
            data_pack.data_type = 'test'
            data_handler.left_points = (minus_data.shape[-1] - data_handler.window_len) % data_handler.slide_len
            del minus_data, minus_label, raw_data_loader, raw_label_loader

            # data_handler.window_len = int(2 * data_handler.sample_rate)
            # data_handler.slide_len = int(1 * data_handler.sample_rate)
            data_handler.window_len = int(1 * data_handler.sample_rate)
            data_handler.slide_len = int(1 * data_handler.sample_rate)

        seg_data_pack = data_handler.get_segment_data(data_pack)

        main_data = seg_data_pack.data
        main_label = seg_data_pack.true_minus_label
        main_ch_loc = seg_data_pack.ch_loc
        main_dataset = EEGBatchData(
            data_handler,
            main_data,
            main_label,
            main_ch_loc,
            data_type,
            n_process_loader,
            cal_timeShift_label,
        )

        if data_type == 'ssl':
            data_pack = data_handler.get_valid_clean_data(ch_list=channel_list, normalize=normalize)
        elif data_type == 'train':
            data_pack = data_handler.get_valid_data(ch_list=channel_list, normalize=normalize)
        else:
            return main_dataset

        seg_data_pack = data_handler.get_segment_data(data_pack)

        valid_data = seg_data_pack.data
        valid_label = seg_data_pack.true_minus_label
        valid_ch_loc = seg_data_pack.ch_loc
        valid_dataset = EEGBatchData(
            data_handler,
            valid_data,
            valid_label,
            valid_ch_loc,
            data_type,
            n_process_loader,
            cal_timeShift_label,
        )
        return main_dataset, valid_dataset

    else:
        main_data_handler_list = []
        main_data_list = []
        main_label_list = []
        valid_data_handler_list = []
        valid_data_list = []
        valid_label_list = []

        for patient in patient_list:
            database_dir = os.path.join(database_save_dir, patient)
            data_dir = os.path.join(data_save_dir, patient)
            data_handler = DataHandler(
                database_save_dir=database_dir,
                data_save_dir=data_dir,
                window_time=window_time,
                slide_time=slide_time,
            )
            data_handler.multi_level = multi_level
            data_handler.shared_encoder = shared_encoder

            assert data_type in ['ssl', 'train', 'test', 'infer']
            if data_type == 'ssl':
                data_pack = data_handler.get_clean_data(ch_list=channel_list, normalize=normalize)
            elif data_type == 'train':
                data_pack = data_handler.get_train_data(ch_list=channel_list, normalize=normalize)

            seg_data_pack = data_handler.get_segment_data(data_pack)

            main_data_handler_list.append(data_handler)
            main_data_list.append(seg_data_pack.data)
            main_label_list.append(seg_data_pack.true_minus_label)

            if data_type == 'ssl':
                data_pack = data_handler.get_valid_clean_data(ch_list=channel_list, normalize=normalize)
            # elif data_type == 'train':
            #     data_pack = data_handler.get_valid_data(ch_list=channel_list, normalize=normalize)

            seg_data_pack = data_handler.get_segment_data(data_pack)

            valid_data_handler_list.append(data_handler)
            valid_data_list.append(seg_data_pack.data)
            valid_label_list.append(seg_data_pack.true_minus_label)

        main_dataset = EEGBatchData_Cross(
            main_data_handler_list,
            main_data_list,
            main_label_list,
            data_type,
            n_process_loader,
            cal_timeShift_label,
        )

        if data_type == 'ssl':
            valid_dataset = EEGBatchData_Cross(
                valid_data_handler_list,
                valid_data_list,
                valid_label_list,
                data_type,
                n_process_loader,
                cal_timeShift_label,
            )

        elif data_type == 'train':
            database_dir = os.path.join(database_save_dir, valid_patient)
            data_dir = os.path.join(data_save_dir, valid_patient)
            data_handler = DataHandler(
                database_save_dir=database_dir,
                data_save_dir=data_dir,
                window_time=window_time,
                slide_time=slide_time,
            )
            data_handler.multi_level = multi_level
            data_handler.shared_encoder = shared_encoder

            data_pack = data_handler.get_valid_data(ch_list=channel_list, normalize=normalize)
            seg_data_pack = data_handler.get_segment_data(data_pack)

            valid_data = seg_data_pack.data
            valid_label = seg_data_pack.true_minus_label
            valid_ch_loc = seg_data_pack.ch_loc
            valid_dataset = EEGBatchData(
                data_handler,
                valid_data,
                valid_label,
                valid_ch_loc,
                data_type,
                n_process_loader,
                cal_timeShift_label,
            )

        return main_dataset, valid_dataset

class CrossSampler(Sampler):
    def __init__(self, data_source, batch_size):
        # super(CrossSampler, self).__init__()
        self.total_length = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        first_index = torch.randperm(self.total_length//self.batch_size)
        first_index *= self.batch_size
        all_index = [first_index]
        for i in range(1, self.batch_size):
            all_index.append(first_index + i)
        all_index = torch.stack(all_index, dim=1).view(-1)
        return iter(all_index.tolist())

    def __len__(self):
        return self.total_length

class EEGBatchData_Cross(Dataset):
    def __init__(
            self,
            data_handler, # list
            data, # list
            label, # list
            data_type,
            n_process_loader=50,
            cal_timeShift_label=None,
    ):
        # list
        self.data_handler = data_handler
        # data.size(): SegNum * ChannelNum * WindowSize (list)
        self.data = [torch.tensor(patient_data, dtype=torch.float) for patient_data in data]
        self.data_size = sum([sub_data.size(0) for sub_data in self.data])
        self.sub_data_size = self.data[0].size(0)
        # label.size(): SegNum * ChannelNum (list)
        self.label = [torch.tensor(patient_label, dtype=torch.long) for patient_label in label]
        # self.ch_loc = torch.tensor(ch_loc, dtype=torch.long) if ch_loc is not None else None
        self.data_type = data_type
        self.nProcessLoader = n_process_loader

        self.reload_pool = Pool(n_process_loader)

        self.n_class = 0
        for i in range(len(self.label)):
            self.n_class = max(self.n_class, len(self.label[i].unique()))
        if (self.data_type != 'infer' and self.n_class > 2) or (self.data_type == 'infer' and self.n_class > 3):
            raise ValueError('There exist illegal labels in the', data_type, 'database.')

        sample_ratio = 0
        for i in range(len(self.label)):
            sample_ratio += (self.label[i] == 1).sum().float() / (self.label[i] != 2).sum().float()
        sample_ratio /= len(self.label)
        print('Channel class weight is: ', torch.tensor([sample_ratio, 1 - sample_ratio]))

        self.multi_level = data_handler[0].multi_level
        self.shared_encoder = data_handler[0].shared_encoder
        self.seg_num = data_handler[0].seg_num
        # self.brain_dict = None
        # self.brain_num = 1
        # if (self.data_type == 'ssl' and not self.shared_encoder) or \
        #         (self.data_type != 'ssl' and (self.multi_level or not self.shared_encoder)):
        #     with open(os.path.join(data_handler.data_save_dir, 'brain_dict.pkl'), 'rb') as f:
        #         self.brain_dict = pickle.load(f)
        #         self.brain_num = len(self.brain_dict)

        # self.brain_label = None
        # self.patient_label = None
        # if self.multi_level and self.data_type != 'ssl':
        #     self.combine_data()

        # whether precalculate time shift label
        self.cal_timeShift_label = True if cal_timeShift_label else False
        if cal_timeShift_label:
            self.time_shift_method = 'sample_idx'
            self.time_shift_measure_steps = cal_timeShift_label
            # choices = ['predefined_idx', 'predefined_score']
            self.time_shift_label = [[] for _ in range(len(self.data))]
            # print('-' * 20, 'pre calculate time-shift label', '-' * 20)
            self.preCal_timeShift()


    def preCal_timeShift(self):
        for k in range(len(self.data)):
            mini_batch, channel_num, dim = self.data[k].size()
            crossCorrelation = torch.nn.CosineSimilarity(dim=-1)
            measure_steps = self.time_shift_measure_steps
            sample_rate = 0.15

            if self.time_shift_method == 'predefined_idx':
                sample_num = int(sample_rate * measure_steps * channel_num**2)

                for batch_idx in range(mini_batch//self.seg_num):
                    batch_data = self.data[batch_idx*self.seg_num:(batch_idx+1)*self.seg_num]
                    # batch_data.size(): seg_num * channel_num * dim
                    batch_idx_list = [[] for _ in range(4)]
                    for i in range(self.seg_num - measure_steps):
                        source = torch.repeat_interleave(batch_data[i], measure_steps * channel_num, dim=0)
                        target = batch_data[i + 1:i + 1 + measure_steps].view(-1, batch_data.size(-1)).repeat(channel_num, 1)
                        crossCor = crossCorrelation(source, target)
                        _, indices = torch.sort(crossCor, descending=True)
                        idx1 = indices // (measure_steps * channel_num)
                        idx2 = (indices % (measure_steps * channel_num)) // channel_num
                        idx3 = (indices % (measure_steps * channel_num)) % channel_num

                        idx0 = torch.tensor([i for _ in range(sample_num*2)])
                        idx1 = torch.cat((idx1[:sample_num], idx1[-sample_num:]))
                        idx2 = torch.cat((idx2[:sample_num], idx2[-sample_num:]))
                        idx3 = torch.cat((idx3[:sample_num], idx3[-sample_num:]))

                        batch_idx_list[0].append(idx0)
                        batch_idx_list[1].append(idx1)
                        batch_idx_list[2].append(idx2)
                        batch_idx_list[3].append(idx3)

                    combine_list = []
                    for idx_num in range(4):
                        combine_list.append(torch.cat(batch_idx_list[idx_num]))
                    combine_idx = torch.stack(combine_list, dim=0)
                    self.time_shift_label.append(combine_idx)

            elif self.time_shift_method == 'predefined_score':
                for batch_idx in range(mini_batch//self.seg_num):
                    batch_data = self.data[batch_idx*self.seg_num:(batch_idx+1)*self.seg_num]
                    # batch_data.size(): seg_num * channel_num * dim
                    mini_batch = []
                    for i in range(self.seg_num - measure_steps):
                        source = torch.repeat_interleave(batch_data[i], measure_steps * channel_num, dim=0)
                        target = batch_data[i + 1:i + 1 + measure_steps].view(-1, batch_data.size(-1)).repeat(channel_num, 1)
                        crossCor = crossCorrelation(source, target)
                        mini_batch.append(crossCor)
                    self.time_shift_label.append(torch.stack(mini_batch, dim=0))

            elif self.time_shift_method == 'sample_idx':
                for batch_idx in range(mini_batch//self.seg_num):
                    batch_data = self.data[k][batch_idx * self.seg_num:(batch_idx + 1) * self.seg_num]
                    # batch_data.size(): seg_num * channel_num * dim
                    source_batch = []
                    target_batch = []
                    for i in range(self.seg_num - measure_steps):
                        source = torch.repeat_interleave(batch_data[i], measure_steps * channel_num, dim=0)
                        target = batch_data[i + 1:i + 1 + measure_steps].view(-1, batch_data.size(-1)).repeat(channel_num, 1)
                        source_batch.append(source)
                        target_batch.append(target)
                    source_batch = torch.stack(source_batch, dim=0)
                    target_batch = torch.stack(target_batch, dim=0)

                    cross_cor = crossCorrelation(source_batch, target_batch)
                    self.time_shift_label[k].append(cross_cor)




    # def combine_data(self):
    #     # We have cut the segments rather than sampling
    #     # Not use .keys() to guarantee the order of brain regions because other codes will use .items()
    #     self.brain_label = []
    #     for _, brain_list in self.brain_dict.items():
    #         self.brain_label.append(self.label[:, brain_list].max(dim=-1, keepdim=True)[0])
    #     # brain_label.size(): SegNum * BrainNum
    #     self.brain_label = torch.cat(self.brain_label, dim=-1)
    #
    #     sample_ratio = (self.brain_label == 1).sum().float() / (self.brain_label != 2).sum().float()
    #     print('Brain class weight is: ', torch.tensor([sample_ratio, 1 - sample_ratio]))
    #
    #     # patient_label.size(): SegNum * 1
    #     self.patient_label = self.brain_label.max(dim=-1, keepdim=True)[0]
    #
    #     sample_ratio = (self.patient_label == 1).sum().float() / (self.patient_label != 2).sum().float()
    #     print('Patient class weight is: ', torch.tensor([sample_ratio, 1 - sample_ratio]))

    def __len__(self):
        # The length is for big segments not for small segments
        return self.data_size // self.seg_num

    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise ValueError('Index out of bound: ', index)

        # When the self.seg_num == 1, torch.tensor[x:x+1] will keep the first dimension as 1,
        # so we should remove the extra dimension
        if index < (self.sub_data_size // self.seg_num):
            out_data = self.data[0][index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
            out_label = self.label[0][index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)

            if self.cal_timeShift_label:
                time_shift_idx = self.time_shift_label[0][index]
                return out_data, out_label, time_shift_idx

        elif (self.sub_data_size // self.seg_num) <= index < (self.sub_data_size*2 // self.seg_num):
            index = index - (self.sub_data_size // self.seg_num)
            out_data = self.data[1][index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
            out_label = self.label[1][index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)

            if self.cal_timeShift_label:
                time_shift_idx = self.time_shift_label[1][index]
                return out_data, out_label, time_shift_idx

        else:
            index = index - (self.sub_data_size*2 // self.seg_num)
            out_data = self.data[2][index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
            out_label = self.label[2][index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)

            if self.cal_timeShift_label:
                time_shift_idx = self.time_shift_label[2][index]
                return out_data, out_label, time_shift_idx

        # if self.multi_level:
        #     out_brain_label = self.brain_label[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
        #     out_patient_label = self.patient_label[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
        #     return out_data, out_label, out_brain_label, out_patient_label
        # elif self.data_type == 'ssl':
        #     out_ch_loc = self.ch_loc[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
        #     return out_ch_loc, out_data, out_label
        return out_data, out_label

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)

    def get_cross_data_loader(self, batch_size, num_workers=0):
        modify_sampler = CrossSampler(self, batch_size)
        return DataLoader(self,
                          batch_size = batch_size,
                          num_workers=num_workers,
                          sampler=modify_sampler,
                          )

class EEGBatchData(Dataset):
    def __init__(
            self,
            data_handler,
            data,
            label,
            ch_loc,
            data_type,
            n_process_loader=50,
            cal_timeShift_label=None,
    ):
        self.data_handler = data_handler
        # data.size(): SegNum * ChannelNum * WindowSize
        self.data = torch.tensor(data, dtype=torch.float)
        # label.size(): SegNum * ChannelNum
        self.label = torch.tensor(label, dtype=torch.long)
        self.ch_loc = torch.tensor(ch_loc, dtype=torch.long) if ch_loc is not None else None
        self.data_type = data_type
        self.nProcessLoader = n_process_loader

        self.reload_pool = Pool(n_process_loader)
        self.n_class = len(self.label.unique())
        if (self.data_type != 'infer' and self.n_class > 2) or (self.data_type == 'infer' and self.n_class > 3):
            raise ValueError('There exist illegal labels in the', data_type, 'database.')
        sample_ratio = (self.label == 1).sum().float() / (self.label != 2).sum().float()
        print('Channel class weight is: ', torch.tensor([sample_ratio, 1 - sample_ratio]))

        self.multi_level = data_handler.multi_level
        self.shared_encoder = data_handler.shared_encoder
        self.seg_num = data_handler.seg_num
        self.brain_dict = None
        self.brain_num = 1
        if (self.data_type == 'ssl' and not self.shared_encoder) or \
                (self.data_type != 'ssl' and (self.multi_level or not self.shared_encoder)):
            with open(os.path.join(data_handler.data_save_dir, 'brain_dict.pkl'), 'rb') as f:
                self.brain_dict = pickle.load(f)
                self.brain_num = len(self.brain_dict)

        self.brain_label = None
        self.patient_label = None
        if self.multi_level and self.data_type != 'ssl':
            self.combine_data()

        # whether precalculate time shift label
        self.cal_timeShift_label = True if cal_timeShift_label else False
        if cal_timeShift_label:
            self.time_shift_method = 'sample_idx'
            self.time_shift_measure_steps = cal_timeShift_label
            # choices = ['predefined_idx', 'predefined_score']
            self.time_shift_label = []
            # print('-' * 20, 'pre calculate time-shift label', '-' * 20)
            self.preCal_timeShift()


    def preCal_timeShift(self):
        mini_batch, channel_num, dim = self.data.size()
        crossCorrelation = torch.nn.CosineSimilarity(dim=-1)
        measure_steps = self.time_shift_measure_steps
        sample_rate = 0.15

        if self.time_shift_method == 'predefined_idx':
            sample_num = int(sample_rate * measure_steps * channel_num**2)

            for batch_idx in range(mini_batch//self.seg_num):
                batch_data = self.data[batch_idx*self.seg_num:(batch_idx+1)*self.seg_num]
                # batch_data.size(): seg_num * channel_num * dim
                batch_idx_list = [[] for _ in range(4)]
                for i in range(self.seg_num - measure_steps):
                    source = torch.repeat_interleave(batch_data[i], measure_steps * channel_num, dim=0)
                    target = batch_data[i + 1:i + 1 + measure_steps].view(-1, batch_data.size(-1)).repeat(channel_num, 1)
                    crossCor = crossCorrelation(source, target)
                    _, indices = torch.sort(crossCor, descending=True)
                    idx1 = indices // (measure_steps * channel_num)
                    idx2 = (indices % (measure_steps * channel_num)) // channel_num
                    idx3 = (indices % (measure_steps * channel_num)) % channel_num

                    idx0 = torch.tensor([i for _ in range(sample_num*2)])
                    idx1 = torch.cat((idx1[:sample_num], idx1[-sample_num:]))
                    idx2 = torch.cat((idx2[:sample_num], idx2[-sample_num:]))
                    idx3 = torch.cat((idx3[:sample_num], idx3[-sample_num:]))

                    batch_idx_list[0].append(idx0)
                    batch_idx_list[1].append(idx1)
                    batch_idx_list[2].append(idx2)
                    batch_idx_list[3].append(idx3)

                combine_list = []
                for idx_num in range(4):
                    combine_list.append(torch.cat(batch_idx_list[idx_num]))
                combine_idx = torch.stack(combine_list, dim=0)
                self.time_shift_label.append(combine_idx)

        elif self.time_shift_method == 'predefined_score':
            for batch_idx in range(mini_batch//self.seg_num):
                batch_data = self.data[batch_idx*self.seg_num:(batch_idx+1)*self.seg_num]
                # batch_data.size(): seg_num * channel_num * dim
                mini_batch = []
                for i in range(self.seg_num - measure_steps):
                    source = torch.repeat_interleave(batch_data[i], measure_steps * channel_num, dim=0)
                    target = batch_data[i + 1:i + 1 + measure_steps].view(-1, batch_data.size(-1)).repeat(channel_num, 1)
                    crossCor = crossCorrelation(source, target)
                    mini_batch.append(crossCor)
                self.time_shift_label.append(torch.stack(mini_batch, dim=0))

        elif self.time_shift_method == 'sample_idx':
            for batch_idx in range(mini_batch//self.seg_num):
                batch_data = self.data[batch_idx * self.seg_num:(batch_idx + 1) * self.seg_num]
                # batch_data.size(): seg_num * channel_num * dim
                source_batch = []
                target_batch = []
                for i in range(self.seg_num - measure_steps):
                    source = torch.repeat_interleave(batch_data[i], measure_steps * channel_num, dim=0)
                    target = batch_data[i + 1:i + 1 + measure_steps].view(-1, batch_data.size(-1)).repeat(channel_num, 1)
                    source_batch.append(source)
                    target_batch.append(target)
                source_batch = torch.stack(source_batch, dim=0)
                target_batch = torch.stack(target_batch, dim=0)

                cross_cor = crossCorrelation(source_batch, target_batch)
                self.time_shift_label.append(cross_cor)




    def combine_data(self):
        # We have cut the segments rather than sampling
        # Not use .keys() to guarantee the order of brain regions because other codes will use .items()
        self.brain_label = []
        for _, brain_list in self.brain_dict.items():
            self.brain_label.append(self.label[:, brain_list].max(dim=-1, keepdim=True)[0])
        # brain_label.size(): SegNum * BrainNum
        self.brain_label = torch.cat(self.brain_label, dim=-1)

        sample_ratio = (self.brain_label == 1).sum().float() / (self.brain_label != 2).sum().float()
        print('Brain class weight is: ', torch.tensor([sample_ratio, 1 - sample_ratio]))

        # patient_label.size(): SegNum * 1
        self.patient_label = self.brain_label.max(dim=-1, keepdim=True)[0]

        sample_ratio = (self.patient_label == 1).sum().float() / (self.patient_label != 2).sum().float()
        print('Patient class weight is: ', torch.tensor([sample_ratio, 1 - sample_ratio]))

    def __len__(self):
        # The length is for big segments not for small segments
        return self.data.size(0) // self.seg_num

    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise ValueError('Index out of bound: ', index)

        # When the self.seg_num == 1, torch.tensor[x:x+1] will keep the first dimension as 1,
        # so we should remove the extra dimension
        out_data = self.data[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
        out_label = self.label[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)

        if self.cal_timeShift_label:
            time_shift_idx = self.time_shift_label[index]
            return out_data, out_label, time_shift_idx

        if self.multi_level:
            out_brain_label = self.brain_label[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
            out_patient_label = self.patient_label[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
            return out_data, out_label, out_brain_label, out_patient_label
        # elif self.data_type == 'ssl':
        #     out_ch_loc = self.ch_loc[index * self.seg_num:(index + 1) * self.seg_num].squeeze(dim=0)
        #     return out_ch_loc, out_data, out_label
        return out_data, out_label

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)
