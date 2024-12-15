import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(
            self,
            data_dir='/data/EEG_Emotion_Recognition/',
            split='train'
    ):
        self.split = split
        self.save_dir = data_dir
        if split == 'train':
            self.data = np.load(os.path.join(self.save_dir, 'ssl_train_data.npz'))['arr_0']
            self.data = torch.tensor(self.data, dtype=torch.float)
        elif split == 'dev':
            self.data = np.load(os.path.join(self.save_dir, 'ssl_val_data.npz'))['arr_0']
            self.data = torch.tensor(self.data, dtype=torch.float)

        self.time_shift_measure_steps = 7
        self.time_shift_label = []
        self.seg_num = self.data.shape[1]

        self.cal_timeShift_label = True
        if self.cal_timeShift_label:
            self.preCal_timeShift()

    def __len__(self):
        return self.data.size(0)

    def preCal_timeShift(self):
        batch_size, time_span, channel_num, dim = self.data.size()
        crossCorrelation = torch.nn.CosineSimilarity(dim=-1)
        measure_steps = self.time_shift_measure_steps

        for batch_idx in range(batch_size):
            batch_data = self.data[batch_idx]
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

    def __getitem__(self, idx):
        out_data = self.data[idx]
        return out_data, self.time_shift_label[idx]


def load_dataset_ssl(
        data_dir,
        train_batch_size,
        test_batch_size,
        num_workers=8
):
    dataloaders = {}
    datasets = {}

    for split in ['train', 'dev']:

        dataset = EmotionDataset(data_dir=data_dir,
                                 split=split)
        if split == 'train':
            shuffle = True
            batch_size = train_batch_size
        else:
            shuffle = False
            batch_size = test_batch_size

        loader = DataLoader(dataset=dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)
        dataloaders[split] = loader
        datasets[split] = dataset

    return dataloaders, datasets
