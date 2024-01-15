import argparse
import numpy as np
import os
import sys
import pickle
import random

from .d_data_file_load import RawDataLoader, RawLabelLoader


def extract_database(
        root_path,              # The root path of the patient's files
        file_name_list,         # The name list of the patient's files
        sample_index,           # The sample index list of every file of the patient
        all_label,
        save_dir,               # The save file holder of the database
        test_file_index,        # The only file index left out for testing
        test_database=False,    # The flag of whether to generate sampling test database
        seg=1000,               # The total segments of train stage of main model
        seg_len=20000,          # The long segment length of main model
        train_ratio=0.8,        # The train ratio of total segments of main model
        normal_ratio=0.5,       # The normal data ratio in train and valid segments of main model
        test_seg=None,          # The list of total test segments of main model
        test_normal_seg=None,   # The list of normal test segments of main model
        clean_seg=10000,        # The total segments of SSL stage
        clean_seg_len=2000,     # The segment length of SSL model
        clean_train_ratio=0.9,  # The train ratio of total SSL segments
):
    # Default test ratios: 1:5; 1:50; 1:500
    if test_seg is None:
        test_seg = [60, 510, 5010]
        test_normal_seg = [50, 500, 5000]

    # Divide the train and test data file
    if test_file_index is None:
        raise ValueError('The test file name does not exist:', test_file_index)
    print('The test file index is:', test_file_index)
    test_file_name_list = [file_name_list[test_file_index]]
    train_file_name_list = file_name_list[:test_file_index] + file_name_list[test_file_index + 1:]
    test_sample_index = [sample_index[test_file_index]]
    train_sample_index = sample_index[:test_file_index] + sample_index[test_file_index + 1:]
    test_all_label = [all_label[test_file_index]]
    train_all_label = all_label[:test_file_index] + all_label[test_file_index + 1:]
    del file_name_list, sample_index, all_label

    print('-'*10, 'Decompose the train files', '-'*10)
    total_all_file_list, total_anomaly_file_list, total_normal_file_list, \
        total_all_file_num_list, total_anomaly_file_num_list, total_normal_file_num_list, \
        total_all_start_num_list, total_anomaly_start_num_list, total_normal_start_num_list = \
        __decompose_data__(
            label=train_all_label,
            seg_len=seg_len,
            clean_seg_len=clean_seg_len,
        )

    print('-'*10, 'Sampling the self-supervised learning database', '-'*10)
    minus_channel_name_path = os.path.join(root_path, 'minus_label_dict.pkl')
    with open(minus_channel_name_path, 'rb') as f:
        channel_num = len(pickle.load(f))
        print('Channel number of the patient is: ', channel_num)
        f.close()
    clean_list, valid_clean_list = __sampling_segment__(
        total_all_file_num_list,
        total_all_start_num_list,
        seg=clean_seg,
        seg_len=clean_seg_len,
        train_ratio=clean_train_ratio,
        valid_ratio=1 - clean_train_ratio,
    )

    print('-' * 10, 'Sampling the training anomaly database of main model', '-' * 10)
    train_anomaly_list, valid_anomaly_list = __sampling_segment__(
        total_anomaly_file_num_list,
        total_anomaly_start_num_list,
        seg=int(seg * (1 - normal_ratio)),
        seg_len=seg_len,
        train_ratio=train_ratio,
        valid_ratio=1 - train_ratio,
    )

    print('-' * 10, 'Sampling the training normal database of main model', '-' * 10)
    train_normal_list, valid_normal_list = __sampling_segment__(
        total_normal_file_num_list,
        total_normal_start_num_list,
        seg=int(seg * normal_ratio),
        seg_len=seg_len,
        train_ratio=train_ratio,
        valid_ratio=1 - train_ratio,
    )

    print('-' * 10, 'Saving training data of SSL and main model', '-' * 10)
    __save_data__(total_all_file_list, clean_list, total_all_file_list, clean_list, root_path,
                  train_file_name_list, train_sample_index, train_all_label, channel_num, save_dir, 'clean')
    __save_data__(total_all_file_list, valid_clean_list, total_all_file_list, valid_clean_list, root_path,
                  train_file_name_list, train_sample_index, train_all_label, channel_num, save_dir, 'valid_clean')
    __save_data__(total_normal_file_list, train_normal_list, total_anomaly_file_list, train_anomaly_list, root_path,
                  train_file_name_list, train_sample_index, train_all_label, channel_num, save_dir, 'train')
    __save_data__(total_normal_file_list, valid_normal_list, total_anomaly_file_list, valid_anomaly_list, root_path,
                  train_file_name_list, train_sample_index, train_all_label, channel_num, save_dir, 'valid')

    if test_database:
        print('-' * 10, 'Decompose the test files', '-' * 10)
        _, total_anomaly_file_list, total_normal_file_list, \
            _, total_anomaly_file_num_list, total_normal_file_num_list, \
            _, total_anomaly_start_num_list, total_normal_start_num_list = \
            __decompose_data__(
                label=test_all_label,
                seg_len=seg_len,
                clean_seg_len=clean_seg_len,
            )

        print('-' * 10, 'Sampling the testing anomaly database of main model', '-' * 10)
        test_anomaly_list = __sampling_segment__(
            total_anomaly_file_num_list,
            total_anomaly_start_num_list,
            seg_len=seg_len,
            test_seg=[test_seg[i] - test_normal_seg[i] for i in range(len(test_seg))],
        )

        print('-' * 10, 'Sampling the testing normal database of main model', '-' * 10)
        test_normal_list = __sampling_segment__(
            total_normal_file_num_list,
            total_normal_start_num_list,
            seg_len=seg_len,
            test_seg=test_normal_seg,
        )

        print('-' * 10, 'Saving testing data of main model', '-' * 10)
        for seg_num in range(len(test_normal_list)):
            ratio = test_normal_seg[seg_num] // (test_seg[seg_num] - test_normal_seg[seg_num])
            __save_data__(total_normal_file_list, test_normal_list[seg_num], total_anomaly_file_list,
                          test_anomaly_list[seg_num], root_path, test_file_name_list, test_sample_index, test_all_label,
                          channel_num, save_dir, 'test_ratio' + str(ratio))


# Extract the valid (for SSL), normal, abnormal (for main model) data
def __decompose_data__(
        label,
        seg_len=20000,
        clean_seg_len=2000,
):
    # Record the cut list of every file
    total_all_file_list = []
    total_anomaly_file_list = []
    total_normal_file_list = []
    # Record the cut numbers of every file
    total_all_file_num_list = []
    total_anomaly_file_num_list = []
    total_normal_file_num_list = []
    # Record the candidate start points number list of every file
    total_all_start_num_list = []
    total_anomaly_start_num_list = []
    total_normal_start_num_list = []
    # Record the anomaly cut and the invalid cut for normal stage
    total_drop_file_list = []
    # Record the total length of every file
    file_total_len = []

    # First: extract all the invalid cuts and total length
    for file_num in range(len(label)):
        total_drop_file = []
        # Find the invalid cut
        label_index = np.where(label[file_num] < 0)[0]
        # Have at least one invalid cut
        if len(label_index) != 0:
            slot_pair_index = __compute_slot__(label_index)
            for s, e in slot_pair_index:
                start = label_index[s]
                end = label_index[e]
                # end + 1 is for the slice operation: [x:y] y not include
                total_drop_file.append((start, end + 1))
        total_drop_file_list.append(total_drop_file)
        file_total_len.append(len(label[file_num]))

    print('total_invalid_file_list:\n', total_drop_file_list)
    print('file_total_len:\n', file_total_len)

    # Second: extract all the ssl start sampling points
    # Record the gap segments between the invalid data
    for file_num in range(len(label)):
        total_all_file = []
        total_all_file_num = 0
        total_all_start_num = []
        last_end = 0
        for (s, e) in total_drop_file_list[file_num] + [(file_total_len[file_num], file_total_len[file_num])]:
            if s < last_end:
                print(total_drop_file_list[file_num])
                raise ValueError('There exist two overlapping invalid segments in file NO.', file_num)
            if s - last_end >= clean_seg_len:
                total_all_file.append((last_end, s - clean_seg_len + 1))
                total_all_start_num.append(s - clean_seg_len + 1 - last_end)
                total_all_file_num += 1
            last_end = e
        total_all_file_list.append(total_all_file)
        total_all_start_num_list.append(total_all_start_num)
        total_all_file_num_list.append(total_all_file_num)

    print('total_all_file_list:\n', total_all_file_list)
    print('total_all_file_num_list:\n', total_all_file_num_list)
    print('total_all_start_num_list:\n', total_all_start_num_list)

    # Third: extract all the anomaly start sampling points
    for file_num in range(len(label)):
        total_anomaly_file = []
        total_anomaly_file_num = 0
        total_anomaly_start_num = []

        # Find the anomaly cuts
        label_index = np.where(label[file_num] == 1)[0]
        # Have at least one anomaly cut
        if len(label_index) != 0:
            slot_pair_index = __compute_slot__(label_index)
            # Merge the continuous epileptic segments to verify
            # there will not exist abnormal labels in normal segments.
            start_index_list, end_index_list = [], []
            pair_index = 0
            while pair_index < len(slot_pair_index):
                start_index = slot_pair_index[pair_index][0]
                end_index = slot_pair_index[pair_index][1]

                end = label_index[end_index]
                look_backward = end + seg_len

                while look_backward != end:
                    # To find whether there are continuous abnormal segments
                    # Avoid to be out of the end file bound
                    if look_backward > len(label[file_num]):
                        look_backward = len(label[file_num])
                    t = np.nonzero(label[file_num][end + 1:look_backward])[0]
                    pair_index += 1
                    if len(t) == 0 or label[file_num][end + 1:look_backward][t.min()] != 1:
                        look_backward = end
                    else:
                        end_index = slot_pair_index[pair_index][1]
                        end = label_index[end_index]
                        look_backward = end + seg_len
                start_index_list.append(start_index)
                end_index_list.append(end_index)
            slot_pair_index = list(zip(start_index_list, end_index_list))

            for s, e in slot_pair_index:
                start = label_index[s]
                end = label_index[e]

                look_forward = start - seg_len + 1
                # Stop when there exist some other negative illegal labels
                # Avoid to be out of the start bound
                if look_forward < 0:
                    t = np.nonzero(label[file_num][:start])[0]
                    look_forward = 0 if len(t) == 0 else t.max() + 1
                else:
                    t = np.nonzero(label[file_num][look_forward:start])[0]
                    if len(t) != 0:
                        look_forward += t.max() + 1

                look_backward = end + seg_len
                # Stop when there exist some other negative illegal labels
                # Avoid to be out of the end file bound
                if look_backward > len(label[file_num]):
                    t = np.nonzero(label[file_num][end + 1:])[0]
                    look_backward = len(label[file_num]) if len(t) == 0 else end + t.min() + 1
                else:
                    t = np.nonzero(label[file_num][end + 1:look_backward])[0]
                    if len(t) != 0:
                        look_backward -= seg_len - t.min() - 1

                # Verify the cut has at least one valid start point
                if look_backward - seg_len + 1 - look_forward > 0:
                    total_anomaly_file.append((look_forward, look_backward - seg_len + 1))
                    total_anomaly_start_num.append(look_backward - seg_len + 1 - look_forward)
                    total_anomaly_file_num += 1
                total_drop_file_list[file_num].append((start, end + 1))

        total_anomaly_file_list.append(total_anomaly_file)
        total_anomaly_start_num_list.append(total_anomaly_start_num)
        total_anomaly_file_num_list.append(total_anomaly_file_num)

    print('total_anomaly_file_list:\n', total_anomaly_file_list)
    print('total_anomaly_file_num_list:\n', total_anomaly_file_num_list)
    print('total_anomaly_start_num_list:\n', total_anomaly_start_num_list)

    # Fourth: extract all the normal start sampling points
    for file_num in range(len(label)):
        # NOTICE: We have to resort the `drop_file_list` as the time order, because the first half includes the
        # invalid data and the last half includes the abnormal data, but they are mixed in time order.
        # This will cause the normal segments include the abnormal and negative labels.
        total_drop_file_list[file_num] = sorted(total_drop_file_list[file_num], key=lambda x: x[0])

        # Record the normal cuts for random selection
        total_normal_file = []
        total_normal_file_num = 0
        total_normal_start_num = []
        last_end = 0
        # Record the gap segments between the drop data
        for (s, e) in total_drop_file_list[file_num] + [(file_total_len[file_num], file_total_len[file_num])]:
            if s < last_end:
                print(total_drop_file_list[file_num])
                raise ValueError('There exist two overlapping drop segments in file NO.', file_num)
            if s - last_end >= seg_len:
                total_normal_file.append((last_end, s - seg_len + 1))
                total_normal_start_num.append(s - seg_len + 1 - last_end)
                total_normal_file_num += 1
            last_end = e
        total_normal_file_list.append(total_normal_file)
        total_normal_start_num_list.append(total_normal_start_num)
        total_normal_file_num_list.append(total_normal_file_num)

    print('total_drop_file_list:\n', total_drop_file_list)
    print('total_normal_file_list:\n', total_normal_file_list)
    print('total_normal_file_num_list:\n', total_normal_file_num_list)
    print('total_normal_start_num_list:\n', total_normal_start_num_list)

    return total_all_file_list, total_anomaly_file_list, total_normal_file_list, \
        total_all_file_num_list, total_anomaly_file_num_list, total_normal_file_num_list, \
        total_all_start_num_list, total_anomaly_start_num_list, total_normal_start_num_list


def __compute_slot__(index):
    end_index = np.where(np.diff(index) > 1)[0]
    start_index = end_index + 1
    end_index = np.append(end_index, len(index) - 1)
    start_index = np.append(0, start_index)

    return list(zip(start_index, end_index))


def __sampling_segment__(
        file_num_list,
        start_num_list,
        seg=500,
        seg_len=20000,
        train_ratio=0.8,
        valid_ratio=0.2,
        test_seg=None,
):
    # The real return list
    train_list = []
    valid_list = []
    test_list = []

    # Record the respective cuts and corresponding candidate start points number
    train_seg_list = []
    valid_seg_list = []
    test_seg_list = []
    train_start_num_list = []
    valid_start_num_list = []
    test_start_num_list = []

    if test_seg is None:
        # Randomly select train and valid cuts
        train_random_list = np.random.choice(np.sum(file_num_list),
                                             round(np.sum(file_num_list) * train_ratio),
                                             replace=False)

        cul_seg = 0
        for file_num in range(len(file_num_list)):
            for seg_num in range(file_num_list[file_num]):
                if cul_seg in train_random_list:
                    train_seg_list.append([file_num, seg_num])
                    train_start_num_list.append(start_num_list[file_num][seg_num])
                else:
                    valid_seg_list.append([file_num, seg_num])
                    valid_start_num_list.append(start_num_list[file_num][seg_num])
                cul_seg += 1

        print('train_seg_list:\n', train_seg_list)
        print('train_start_num_list:\n', train_start_num_list)
        print('valid_seg_list:\n', valid_seg_list)
        print('valid_start_num_list:\n', valid_start_num_list)

        # Randomly select training and validating segments
        train_random_list = np.random.choice(np.sum(train_start_num_list),
                                             round(seg * train_ratio))
        for index in train_random_list:
            list_num = 0
            # Scan for which cut
            while index >= train_start_num_list[list_num]:
                index -= train_start_num_list[list_num]
                list_num += 1
            train_list.append([train_seg_list[list_num][0], train_seg_list[list_num][1], index, index + seg_len])
        print('train_list:\n', train_list)

        valid_random_list = np.random.choice(np.sum(valid_start_num_list),
                                             round(seg * valid_ratio))
        for index in valid_random_list:
            list_num = 0
            while index >= valid_start_num_list[list_num]:
                index -= valid_start_num_list[list_num]
                list_num += 1
            valid_list.append([valid_seg_list[list_num][0], valid_seg_list[list_num][1], index, index + seg_len])
        print('valid_list:\n', valid_list)

        return train_list, valid_list
    else:
        # there is a single test file
        for file_num in range(len(file_num_list)):
            for seg_num in range(file_num_list[file_num]):
                test_seg_list.append([file_num, seg_num])
                test_start_num_list.append(start_num_list[file_num][seg_num])

        print('test_seg_list:\n', test_seg_list)
        print('test_start_num_list:\n', test_start_num_list)

        # Extract test dataset for different ratios
        for seg_num in range(len(test_seg)):
            tmp_test_seg = test_seg[seg_num]
            tmp_test_list = []

            test_random_list = np.random.choice(np.sum(test_start_num_list),
                                                tmp_test_seg)
            for index in test_random_list:
                list_num = 0
                while index >= test_start_num_list[list_num]:
                    index -= test_start_num_list[list_num]
                    list_num += 1
                tmp_test_list.append([test_seg_list[list_num][0], test_seg_list[list_num][1], index, index + seg_len])
            test_list.append(tmp_test_list)
        print('test_list:\n', test_list)

        return test_list


def __save_data__(
        normal_total,
        normal_index,
        anomaly_total,
        anomaly_index,
        root_path,
        file_name_list,
        sample_index,
        all_label,
        channel_num,
        save_dir,
        split_type,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    minus_save = []
    normal_minus_save = []
    all_label_save = []
    minus_label_save = []
    loc_save = []
    # loc = [file_num, cut_num, start, end]
    # First aggregate all pairs in one file to reduce the IO read operations
    file_loc_pairs = [[] for _ in range(len(file_name_list))]
    for loc in normal_index:
        s = normal_total[loc[0]][loc[1]][0] + loc[2]
        e = normal_total[loc[0]][loc[1]][0] + loc[3]
        file_loc_pairs[loc[0]].append((s, e))

    clean_channels, ch_count = None, np.inf
    if split_type != 'clean' and split_type != 'valid_clean':
        for loc in anomaly_index:
            s = anomaly_total[loc[0]][loc[1]][0] + loc[2]
            e = anomaly_total[loc[0]][loc[1]][0] + loc[3]
            file_loc_pairs[loc[0]].append((s, e))
    else:
        # Sample the channels
        print('Do not sample channels.')
        # clean_channels = np.random.choice(channel_num, len(normal_index))
        # ch_count = 0
        # print(split_type + ' channels:\n', list(clean_channels))

    for file_num in range(len(file_loc_pairs)):
        loc_pairs = file_loc_pairs[file_num]
        if len(loc_pairs) == 0:
            continue
        file = file_name_list[file_num]
        print(file)
        tmp_sample_index = sample_index[file_num]

        for (s, e) in loc_pairs:
            # Self-check the illegal labels
            if not set(np.unique(all_label[file_num][s:e])).issubset({0, 1}):
                raise ValueError('There exist negative labels in the database from', file, s, e)
            all_label_save.append(all_label[file_num][s:e])
        # all_label have to exist all the time because the _save_data will be called more than once

        minus_data, ori_minus_data = __get_single_file_data__(
            raw_type='data',
            root_path=root_path,
            file=file,
            normalize=False,
            sample_index=tmp_sample_index,
        )
        if clean_channels is None:
            for (s, e) in loc_pairs:
                minus_save.append(minus_data[:, s:e])
        else:
            for (s, e) in loc_pairs:
                minus_save.append(minus_data[clean_channels[ch_count], s:e])
                ch_count += 1
        del minus_data
        ch_count -= len(loc_pairs)

        normal_minus_data, _ = __get_single_file_data__(
            raw_type='data',
            root_path=root_path,
            file=file,
            normalize=True,
            sample_index=tmp_sample_index,
            raw_minus_data=ori_minus_data,
        )
        del ori_minus_data
        if clean_channels is None:
            for (s, e) in loc_pairs:
                normal_minus_save.append(normal_minus_data[:, s:e])
        else:
            for (s, e) in loc_pairs:
                normal_minus_save.append(normal_minus_data[clean_channels[ch_count], s:e])
                ch_count += 1
        del normal_minus_data
        ch_count -= len(loc_pairs)

        minus_label, _ = __get_single_file_data__(
            raw_type='label',
            root_path=root_path,
            file=file,
            normalize=False,
            sample_index=tmp_sample_index,
        )
        if clean_channels is None:
            for (s, e) in loc_pairs:
                # Self-check the illegal labels
                if not set(np.unique(minus_label[:, s:e])).issubset({0, 1}):
                    raise ValueError('There exist negative labels in the database from', file, s, e)
                minus_label_save.append(minus_label[:, s:e])
                loc_save.append((file_num, (s, e)))
        else:
            for (s, e) in loc_pairs:
                if not set(np.unique(minus_label[clean_channels[ch_count], s:e])).issubset({0, 1}):
                    raise ValueError('There exist negative labels in the database from',
                                     file, clean_channels[ch_count], s, e)
                minus_label_save.append(minus_label[clean_channels[ch_count], s:e])
                loc_save.append((file_num, clean_channels[ch_count], (s, e)))
                ch_count += 1
        del minus_label

    np.savez_compressed(os.path.join(save_dir, split_type + '_minus_data.npz'), x=np.array(minus_save))
    np.savez_compressed(os.path.join(save_dir, split_type + '_normal_minus_data.npz'), x=np.array(normal_minus_save))
    np.savez_compressed(os.path.join(save_dir, split_type + '_all_label.npz'), x=np.array(all_label_save))
    np.savez_compressed(os.path.join(save_dir, split_type + '_minus_label.npz'), x=np.array(minus_label_save))
    with open(os.path.join(save_dir, split_type + '_loc.pkl'), 'wb') as f:
        pickle.dump(loc_save, f)


def __get_single_file_data__(
        raw_type,
        root_path,
        file,
        normalize,
        sample_index,
        raw_minus_data=None,
):
    if raw_type == 'data':
        raw_loader = RawDataLoader(
            root_path=root_path,
            file=file,
            normalize=normalize,
        )
        minus_x = raw_loader.minus_data_loader(raw_minus_data=raw_minus_data)
    else:
        raw_loader = RawLabelLoader(
            root_path=root_path,
            file=file,
        )
        minus_x = raw_loader.minus_label_loader()
        # Transform other positive labels to zero labels
        minus_x = np.where(minus_x > 1, 0, minus_x)

    sample_rate = raw_loader.sample_rate
    base_sample_rate = raw_loader.base_sample_rate
    del raw_loader

    # Force all the files in one patient to be the same sample rate
    if sample_rate > base_sample_rate:
        _, sample_minus_x, _, _, _, _ = \
            uniform_sample_data(sample_rate // base_sample_rate, minus_x.shape[-1],
                                sample_index=sample_index, minus_data=minus_x)
    else:
        sample_minus_x = minus_x

    return sample_minus_x, minus_x


def uniform_sample_data(multiple, total_length, sample_index=None, single_data=None, minus_data=None,
                            normal_minus_data=None, all_label=None, minus_label=None):
    if sample_index is None:
        sample_index = []
        scan = random.randint(0, multiple - 1)
        while scan < total_length:
            sample_index.append(scan)
            scan += multiple
    if single_data is not None:
        single_data = single_data[:, sample_index]
    if minus_data is not None:
        minus_data = minus_data[:, sample_index]
    if normal_minus_data is not None:
        normal_minus_data = normal_minus_data[:, sample_index]
    if all_label is not None:
        all_label = all_label[sample_index]
    if minus_label is not None:
        minus_label = minus_label[:, sample_index]
    return single_data, minus_data, normal_minus_data, all_label, minus_label, sample_index


"""
Test files:
05ZLH: 4/13/2018 7:54:27PM——FA0010AP 
06ZYJ: 8/4/2019 3:41:18PM——FA0011MB 
02GJX: 11/20/2019 2:41:40AM——FA00127W 
01TGX: 12/23/2020 3:04:51PM——FA0014KI 
03: 10/18/2020 3:38:38AM——FA00145X 
04: 7/17/2020 1:29:02PM——FA0013OJ 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Database')
    parser.add_argument('--load_dir', type=str, default='/data/eeggroup/new_data2/02GJX/',
                        help='Should give an absolute path including all the original data files of one patient.')
    parser.add_argument('--save_dir', type=str, default='/data/caidonghong/new_database2_no_filter_0612/02GJX/',
                        help='Should give an absolute path to save the database of the patient.')
    parser.add_argument('--test_file', type=str, default='FA00127W',
                        help='Should give a file name saved as the test file in the load_dir.')
    parser.add_argument('--generate_test_database', action='store_false',
                        help='The flag of whether to generate the sampling test database.')
    parser.add_argument('--seg', type=int, default=1000,
                        help='The number of segments to generate.')
    parser.add_argument('--seg_len', type=int, default=10,
                        help='The seconds of every segment to generate.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='The ratio of train segments.')
    parser.add_argument('--normal_ratio', type=float, default=0.5,
                        help='The ratio of normal segments in train and valid segments.')
    parser.add_argument('--test_seg', nargs='*', type=int, default=[60, 510, 5010],
                        help='The number list of test segments to generate.')
    parser.add_argument('--test_normal_seg', nargs='*', type=int, default=[50, 500, 5000],
                        help='The number list of normal test segments to generate.')
    parser.add_argument('--clean_seg', type=int, default=1000,
                        help='The number of SSL segments to generate.')
    parser.add_argument('--clean_seg_len', type=int, default=10,
                        help='The seconds of every SSL segment to generate.')
    parser.add_argument('--clean_train_ratio', type=float, default=0.9,
                        help='The ratio of SSL train segments.')
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    save_dir = args.save_dir
    root_path = args.load_dir

    print('Reading data from:', root_path)
    file_name_list = []
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        if os.path.isdir(file_path):
            file_name_list.append(file)
    print('Total number of files:', len(file_name_list))
    print('All file names:\n', file_name_list)

    all_label, sample_index = [], []
    base_sample_rate = 0
    test_file_index = None
    for i in range(len(file_name_list)):
        file = file_name_list[i]
        print('Reading the file:', file)
        if file == args.test_file:
            test_file_index = i
        raw_label_loader = RawLabelLoader(
            root_path=root_path,
            file=file,
        )

        tmp_all_label = raw_label_loader.all_label_loader()

        tmp_sample_index = []
        base_sample_rate = raw_label_loader.base_sample_rate
        # Force all the files in one patient to be the same sample rate
        if raw_label_loader.sample_rate > raw_label_loader.base_sample_rate:
            _, _, _, tmp_all_label, _, tmp_sample_index = \
                uniform_sample_data(raw_label_loader.sample_rate // raw_label_loader.base_sample_rate,
                                    tmp_all_label.shape[0], all_label=tmp_all_label)

        all_label.append(tmp_all_label)
        sample_index.append(tmp_sample_index)
        del raw_label_loader

    extract_database(
        root_path,
        file_name_list,
        sample_index,
        all_label,
        save_dir,
        test_file_index,
        test_database=args.generate_test_database,
        seg=args.seg,
        seg_len=args.seg_len * base_sample_rate,
        train_ratio=args.train_ratio,
        normal_ratio=args.normal_ratio,
        test_seg=args.test_seg,
        test_normal_seg=args.test_normal_seg,
        clean_seg=args.clean_seg,
        clean_seg_len=args.clean_seg_len * base_sample_rate,
        clean_train_ratio=args.clean_train_ratio,
    )
    print('-' * 10, 'ALL Done', '-' * 10)
