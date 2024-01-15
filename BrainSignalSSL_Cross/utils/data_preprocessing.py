import torch
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from pipelines.g_dataset_dataloader import load_dataset
# from pipeline.g_dataset_dataloader import load_dataset
# from pipeline4EEG.dataloader_ssl import load_dataset_ssl

def draw_one_sec_all(all_data):
    cosSimilarity = torch.nn.CosineSimilarity(dim=1)
    batch_size = len(all_data)
    all_sim_matrix = 0

    for i in range(batch_size):
        data = all_data[i][0]

        source = torch.repeat_interleave(data, data.size(0), dim=0)
        target = data.repeat(data.size(0), 1)
        dot_product = cosSimilarity(source, target)
        sim_matrix = dot_product.reshape(data.size(0), data.size(0))
        all_sim_matrix += sim_matrix

    average_sim_matrix = all_sim_matrix / batch_size
    # plt.figure(figsize=(7, 7), dpi=100, facecolor='w')
    # plt.title("Correlation graph on normal data (idx:%d-%d)\none second" % (0, batch_size))
    # fig = sns.heatmap(average_sim_matrix, annot=False, cmap="Blues", square=True, fmt='.2g')
    return average_sim_matrix

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
    # plt.figure(figsize=(7, 7), dpi=100, facecolor='w')
    # plt.title("Correlation graph on normal data (idx:%d-%d)\none second" % (0, batch_size))
    # fig = sns.heatmap(average_sim_matrix, annot=False, cmap="Blues", square=True, fmt='.2g')
    # plt.show()
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
    # sim_matrix_all = sim_matrix_all - torch.eye(sim_matrix_all.size(0))
    # we take the absolute value of the sim_matrix_all
    # since we think the pair of channel has high negative similarity should be connected in the graph
    # sim_matrix_all = torch.abs(sim_matrix_all)
    del train_dataset

    # idx_x, idx_y = torch.where(sim_matrix_all < threshold)
    # sim_matrix_all[idx_x, idx_y] = 0
    # sim_matrix_all is a matrix only preserve the weight >= threshold, otherwise 0.
    # (elements in diagonal will be replaced with 0, since self-loop will be added in GNN module)
    return sim_matrix_all