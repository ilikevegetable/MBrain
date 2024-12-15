import torch
from pipelines4EEG.dataloader_ssl import load_dataset_ssl

def draw_ten_sec_all(all_data):
    cosSimilarity = torch.nn.CosineSimilarity(dim=-1)
    batch_size = len(all_data)
    all_sim_matrix = 0

    for i in range(batch_size):
        data = all_data[i][0]
        # data.size(): time_span * channel_num * dim
        data = data.permute(1,0,2)
        data = data.reshape(data.size(0), data.size(1)*data.size(2))

        source = torch.repeat_interleave(data, data.size(0), dim=0)
        target = data.repeat(data.size(0), 1)
        dot_product = cosSimilarity(source, target)
        sim_matrix = dot_product.reshape(data.size(0), data.size(0))
        all_sim_matrix += sim_matrix

    average_sim_matrix = all_sim_matrix / batch_size
    return average_sim_matrix

def similarity_mean_eeg(data_dir):
    print('\n', '-'*20,'build coarse-grained correlation matrix','-'*20)
    _, datasets, _ = load_dataset_ssl(
        data_dir=data_dir,
        train_batch_size=16,
        test_batch_size=16,
        num_workers=8,
    )
    sim_matrix_all = draw_ten_sec_all(datasets['train'])
    del datasets
    return sim_matrix_all