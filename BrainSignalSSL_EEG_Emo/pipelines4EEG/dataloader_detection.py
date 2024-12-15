from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os


class EmotionDataset(Dataset):
    def __init__(
            self,
            data_dir='/data/EEG_Emotion_Recognition/',
            split='train'
    ):
        self.save_dir = data_dir
        if split == 'train':
            self.data = np.load(os.path.join(self.save_dir, 'train_data.npz'))['arr_0']
            self.data = torch.tensor(self.data, dtype=torch.float)
            self.label = np.load(os.path.join(self.save_dir, 'train_label.npz'))['arr_0']
            self.label = torch.tensor(self.label, dtype=torch.long)
        elif split == 'dev':
            self.data = np.load(os.path.join(self.save_dir, 'val_data.npz'))['arr_0']
            self.data = torch.tensor(self.data, dtype=torch.float)
            self.label = np.load(os.path.join(self.save_dir, 'val_label.npz'))['arr_0']
            self.label = torch.tensor(self.label, dtype=torch.long)
        elif split == 'test':
            self.data = np.load(os.path.join(self.save_dir, 'test_data.npz'))['arr_0']
            self.data = torch.tensor(self.data, dtype=torch.float)
            self.label = np.load(os.path.join(self.save_dir, 'test_label.npz'))['arr_0']
            self.label = torch.tensor(self.label, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        return out_data, out_label

def load_dataset_detection(
        data_dir,
        train_batch_size,
        test_batch_size=None,
        num_workers=8,
        data_type='train'
):
    dataloaders = {}
    datasets = {}
    if data_type == 'train':
        for split in ['train', 'dev', 'test']:

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
    elif data_type == 'test':
        dataset = EmotionDataset(data_dir=data_dir,
                                 split='test')

        shuffle = False
        batch_size = test_batch_size

        loader = DataLoader(dataset=dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)
        dataloaders['test'] = loader
        datasets['test'] = dataset
    return dataloaders, datasets