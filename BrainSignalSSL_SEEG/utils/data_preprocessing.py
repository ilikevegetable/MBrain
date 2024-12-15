import torch
from pipelines.g_dataset_dataloader import load_dataset


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

def similarity_mean_seeg(database_save_dir, data_save_dir):
    print('\n', '-'*20,'build coarse-grained correlation matrix','-'*20)
    train_dataset = load_dataset(
        database_save_dir=database_save_dir,
        data_save_dir=data_save_dir,
        window_time=1,
        slide_time=1,
        data_type='mean_matrix',
        channel_list=None,
        normalize=True,
        multi_level=False,
        shared_encoder=True,
        n_process_loader=50,
    )
    sim_matrix_all = draw_ten_sec_all(train_dataset)
    del train_dataset

    return sim_matrix_all