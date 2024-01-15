import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import sys
import re
import pickle
from mne.io import read_raw_edf
from collections import OrderedDict


parser = argparse.ArgumentParser(description='LabelProcess')
parser.add_argument('--load_dir', type=str, default='/data/eeggroup/new_data2/02GJX/',
                    help='Should give an absolute path including all .edf files and label files of one patient.')
parser.add_argument('--check', action='store_true',
                    help='The flag of whether to check these label event files. '
                         'One should first set the flag True to check the label files and generate label files.')
argv = sys.argv[1:]
args = parser.parse_args(argv)


# Read all files of the patient in the root path
root_path = args.load_dir
print('Label process path:', root_path)
file_name_list = []
for file in os.listdir(root_path):
    file_path = os.path.join(root_path, file)
    if os.path.isdir(file_path):
        file_name_list.append(file)
print('Total number of files:', len(file_name_list))
print('All file names:\n', file_name_list)


# Transform the natural time to millisecond time
def time_to_millisecond(x):
    time_obj = datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    return int(time.mktime(time_obj.timetuple()) * 1000.0 + time_obj.microsecond / 1000.0)


# Transform to make the error finding easier
def millisecond_to_time(x, s_time):
    x = x + s_time
    time_stamp = float(x / 1000)
    time_array = time.localtime(time_stamp)
    return time.strftime('%Y-%m-%d %H:%M:%S', time_array) + ('%06f' % (time_stamp - int(time_stamp)))[1:]


# The filled values of different special labels
flags = {'S': 1, '?': 2, 'I': 3, 'L': 4, 'G': -1, 'B': -1, 'Cut': -2, 'E': -3, 'P': -4, 'C': -5, 'H': -6}
electrode_file_name = os.path.join(root_path, 'electrode_information.pkl')
with open(electrode_file_name, 'rb') as f:
    # electrode_name: the electrode name list from the original software board
    # electrode_num: the list of channel number of every electrode
    electrode_name, electrode_num = pickle.load(f)

# Record the start and end event of every channel
onset_ch_to_time = {}
# Record the channel names which will be deleted as the bad channels
global_del_channel = []
# Record the sample rate of every file
sample_rate_dict = {}
# Record all the illegal label information and print at the end for check
error_logs = {}
# Record the start time of the electrical stimulus stage
stim_start = None
stim_start_time = None
print('-'*10, 'Start analyzing the label events', '-'*10)
for file in file_name_list:
    print('Processing the file:', file)
    error_logs[file] = []
    data_file_name = os.path.join(root_path, file, file + '.edf')
    event_file_name = os.path.join(root_path, file, file + '.csv')

    raw_data = read_raw_edf(data_file_name, preload=False)
    # The sample rate from .edf files sometimes are not integers
    sample_rate = round(raw_data.info['sfreq'])
    data_length = raw_data[0][0].shape[1]
    print('File sample rate:', sample_rate)
    print('File data point number:', data_length)
    sample_rate_dict[file] = sample_rate
    event = pd.read_csv(event_file_name)
    # ts: True Time; description: Event
    event = event[['ts', 'description']]
    start_time = event.ts[0]
    print('Record start time:', start_time)

    # Remove the space
    event['description'] = event.description.apply(lambda x: x.replace(' ', ''))
    # Compute the time gap from the start event
    event['timestamp'] = event.ts.apply(lambda x: time_to_millisecond(x) - time_to_millisecond(start_time))
    start_time = time_to_millisecond(start_time)
    # location: the real data point index
    event['location'] = np.floor(event['timestamp'] * sample_rate / 1000.0).apply(lambda x: int(x))
    print('Event data frame:\n', event)

    # Record the label process from start to end
    # Every channel has a list and an unique index/count
    # onset_ch_to_time dict format: {channel_name: [index, [record]...]}
    # record format: [flag, start_time_stamp, start_location, onset_state]
    # flag: the value to be filled during a label process. See overall definition of 'flags'.
    # start_location: the start label time point index.
    # onset_state: to record the consistency of label pairs. Specifically, the start label will set this as 'True'
    #              and the end label will set this as 'False'.
    count = 0
    for electrode_index in range(len(electrode_name)):
        # Bipolar leads are one less than the original unipolar leads
        for i in range(1, electrode_num[electrode_index]):
            # Only record the first contact's name of a bipolar lead
            onset_ch_to_time[electrode_name[electrode_index] + str(i)] = [count, [False]]
            count += 1

    if not args.check:
        # Customized Sparse Type (Slice save)
        # [time_length, [triples] * channel_num]
        # triple: [start_index, end_index, value]
        data_label = [data_length, [[]]]
        data_minus_label = [data_length, [[] for _ in range(len(onset_ch_to_time))]]

    # Record whether some events are happening. Usually, we think there should not include any start events after
    # end events in a complete event. If there includes, the start event may be trapped by other extra end events.
    # However, the situations do not always hold. Thus, we should check the error manually.
    event_start_flag = False
    for index, row in event.iterrows():
        if args.check:
            print(row['description'])
        label = re.match(r"([S?ILGBEPCH])/?[\w,;\-']*(#|/end)|(Cut)/[LRB][0-9]*#", row['description'])
        if label is not None:
            # signal: (S,?,I,L,G,B,E,P,C,H; #,/end; Cut) three group tuple
            signal = label.groups()
            if args.check:
                print(signal)

            if signal[1] == '#' or signal[1] == '/end':
                # value to be filled
                flag = flags[signal[0]]
                # Process the electrical stimulus separately because there does not exist channel names
                if signal[0] == 'E':
                    if signal[1] == '#':
                        if stim_start is not None:
                            error_logs[file].append(
                                'Electrical Stimulus not stop at' + millisecond_to_time(stim_start_time, start_time))
                        stim_start = row['location']
                        stim_start_time = row['timestamp']
                    else:
                        if stim_start is None:
                            error_logs[file].append(
                                'Electrical Stimulus not start at' + millisecond_to_time(row['timestamp'], start_time))
                        elif not args.check:
                            data_label[-1][0].append([stim_start, row['location'], flag])
                            # Only assign the zeros to flag to keep the P,C labels
                            for ch_index in range(len(data_minus_label[-1])):
                                insert_index = len(data_minus_label[-1][ch_index]) - 1
                                # [triples]: [insert_index, start_index, end_index] to keep the end index orders
                                insert_entry = []
                                # The real end index of the electrical stimulus
                                tmp_end = row['location']
                                while insert_index >= 0 and \
                                        data_minus_label[-1][ch_index][insert_index][1] >= stim_start:
                                    insert_entry.append(
                                        [insert_index + 1, data_minus_label[-1][ch_index][insert_index][1], tmp_end])
                                    tmp_end = data_minus_label[-1][ch_index][insert_index][0]
                                    insert_index -= 1
                                # The real start index of the electrical stimulus
                                insert_entry.append([insert_index + 1, stim_start, tmp_end])

                                # The insert_entry is recorded as the reverse order, so inserting the tail will not
                                # change the insert index of the former ones
                                for insert_index, start_index, end_index in insert_entry:
                                    data_minus_label[-1][ch_index].insert(insert_index, [start_index, end_index, flag])
                            del insert_entry
                        stim_start = None
                        stim_start_time = None
                # Process the thermocoagulation label. Remove all the subsequent data.
                elif signal[0] == 'H':
                    if not args.check:
                        data_label[-1][0].append([row['location'], data_label[0], flag])
                        for ch_index in range(len(data_minus_label[-1])):
                            data_minus_label[-1][ch_index].append([row['location'], data_minus_label[0], flag])
                    break

                # ch_names: [(electrode_name; electrode_num_range)...] list of two group tuple
                ch_names = re.findall(r"([a-zA-Z]+'?)([0-9,;\-]+)", row['description'])
                if args.check:
                    print(ch_names)

                for ch_num_pair in ch_names:
                    elec_name = ch_num_pair[0]
                    # Here are two candidates: range (x-x) or single (x)
                    # elec_nums: [(num; num/None)...]
                    elec_nums = re.findall(r"([0-9]+)-?([0-9]*)", ch_num_pair[1])
                    if args.check:
                        print(elec_name)
                        print(elec_nums)
                    # record the channels influenced by this label
                    candidate_nums = []
                    for range_num in elec_nums:
                        # single channel
                        if range_num[1] == '':
                            candidate_nums.append(int(range_num[0]))
                        # range channels
                        else:
                            candidate_nums.extend(list(range(int(range_num[0]), int(range_num[1]) + 1)))
                    if args.check:
                        print(candidate_nums)

                    # start label
                    if signal[1] == '#':
                        # Only check this when we first scan the label file, because there exist some exceptions which
                        # are not real errors.
                        if args.check and not event_start_flag:
                            # To avoid some labels running away from the current labeled segment and being trapped in
                            # another segment without start labels but with end labels.
                            for ch_name, state_list in onset_ch_to_time.items():
                                if state_list[-1][-1]:
                                    if state_list[-1][0] == flags['B']:
                                        continue
                                    error_logs[file].append(
                                        'Channel ' + ch_name + ' not stop at ' + millisecond_to_time(
                                            state_list[-1][-3], start_time))
                            event_start_flag = True

                        for num in candidate_nums:
                            ch_name = elec_name + str(num)
                            # If the channel name is not in the channel name list, it's illegal.
                            if ch_name not in onset_ch_to_time.keys():
                                error_logs[file].append(
                                    'Channel ' + ch_name + ' not in the valid channel names list at ' + millisecond_to_time(
                                        row['timestamp'], start_time))
                                continue

                            # If the channel is in the bad state, then any labels will not be recorded
                            if onset_ch_to_time[ch_name][-1][0] in [flags['B'], flags['G']]:
                                continue
                            # If there exists earlier start label of the same channel in this onset stage, the label is
                            # illegal.
                            if onset_ch_to_time[ch_name][-1][-1]:
                                error_logs[file].append('Channel ' + ch_name + ' not stop at ' + millisecond_to_time(
                                    onset_ch_to_time[ch_name][-1][-3],
                                    start_time) + ' and restart at ' + millisecond_to_time(row['timestamp'],
                                                                                           start_time))
                                onset_ch_to_time[ch_name][-1][-1] = False
                            # A new event occurs
                            onset_ch_to_time[ch_name].append([flag, row['timestamp'], row['location'], True])
                            # Process the G/B label - global/local bad channels in one file.
                            if signal[0] == 'G' or signal[0] == 'B':
                                result = re.match(r"([a-zA-Z]+'?)([0-9]+)", ch_name).groups()
                                # Restore the channel name to original bipolar montage name
                                new_key = ch_name + '-' + result[0] + str(int(result[1]) + 1)
                                global_del_channel.append(new_key)

                                # The G label does not have an end one.
                                if signal[0] == 'G':
                                    if not args.check:
                                        data_minus_label[-1][onset_ch_to_time[ch_name][0]].append(
                                            [0, data_minus_label[0], flag])
                                    onset_ch_to_time[ch_name][-1][-1] = False
                    # end label
                    else:
                        for num in candidate_nums:
                            ch_name = elec_name + str(num)
                            # If the channel name is not in the channel name list, it's illegal.
                            if ch_name not in onset_ch_to_time.keys():
                                error_logs[file].append(
                                    'Channel ' + ch_name + ' not in the valid channel names list at ' + millisecond_to_time(
                                        row['timestamp'], start_time))
                                continue

                            # There exists some end labels without corresponding start labels.
                            if onset_ch_to_time[ch_name][-1][-1]:
                                # If the end label's type does not match the start label's, the label is illegal.
                                if onset_ch_to_time[ch_name][-1][0] != flag:
                                    # If the channel is in the bad state, then any labels will not be recorded
                                    if onset_ch_to_time[ch_name][-1][0] == flags['B']:
                                        continue
                                    error_logs[file].append(
                                        'Channel ' + ch_name + ' has a different label flag at ' + millisecond_to_time(
                                            onset_ch_to_time[ch_name][-1][-3],
                                            start_time) + ' and find error at ' + millisecond_to_time(row['timestamp'],
                                                                                                      start_time))
                                    onset_ch_to_time[ch_name][-1][-1] = False
                                else:
                                    if not args.check:
                                        data_minus_label[-1][onset_ch_to_time[ch_name][0]].append(
                                            [onset_ch_to_time[ch_name][-1][-2], row['location'] + 1, flag])
                                        # We regard I/?/L labels as normal data
                                        if signal[0] == 'S':
                                            data_label[-1][0].append(
                                                [onset_ch_to_time[ch_name][-1][-2], row['location'] + 1, flag])
                                    # Set the label's state to be invalid.
                                    onset_ch_to_time[ch_name][-1][-1] = False
                            # If the channel is in the bad state, then any labels will not be recorded
                            elif onset_ch_to_time[ch_name][-1][0] == flags['G']:
                                continue
                            else:
                                error_logs[file].append(
                                    'Channel ' + ch_name + ' not start at ' + millisecond_to_time(row['timestamp'],
                                                                                                  start_time))
                        if args.check:
                            event_start_flag = False
            # cut flag labels
            elif not args.check:
                flag = flags[signal[-1]]
                direc, sec = re.findall(r"([LRB])([0-9]*)", row['description'])[0]

                if direc == 'L':
                    if len(sec) == 0:
                        start_index = 0
                    else:
                        start_index = max(0, row['location'] - int(sec) * sample_rate)
                    end_index = row['location']
                elif direc == 'R':
                    if len(sec) == 0:
                        end_index = data_label[0]
                    else:
                        end_index = min(data_label[0], row['location'] + int(sec) * sample_rate)
                    start_index = row['location']
                else:
                    if len(sec) == 0:
                        raise ValueError('The bidirectional Cut Flag should assign specific seconds.')
                    else:
                        start_index = max(0, row['location'] - int(sec) * sample_rate)
                        end_index = min(data_label[0], row['location'] + int(sec) * sample_rate)

                data_label[-1][0].append([start_index, end_index, flag])
                for ch_index in range(len(data_minus_label[-1])):
                    data_minus_label[-1][ch_index].append([start_index, end_index, flag])

    # Check the electrical stimulus process in case that there does not exist end label.
    if stim_start is not None:
        error_logs[file].append('Electrical Stimulus not stop at' + millisecond_to_time(stim_start_time, start_time))

    for key, value in onset_ch_to_time.items():
        if value[-1][-1]:
            error_logs[file].append(
                'Channel ' + key + ' not stop at ' + millisecond_to_time(value[-1][-3], start_time) + 'in the end')
    if args.check:
        print(onset_ch_to_time)

    if not args.check:
        with open(os.path.join(root_path, file, 'data_minus_label.pkl'), 'wb') as f:
            pickle.dump(data_minus_label, f)
        with open(os.path.join(root_path, file, 'data_label.pkl'), 'wb') as f:
            pickle.dump(data_label, f)
        print('Saving the label files done')
    print('Processing file', file, 'done')
print('-'*10, 'Done', '-'*10)


# Save the bad channels. One bad, all bad.
global_del_channel = list(set(global_del_channel))
print('Bad channels are:\n', global_del_channel)
if not args.check:
    with open(os.path.join(root_path, 'del_channels.pkl'), 'wb') as f:
        pickle.dump(global_del_channel, f)
    print('Saving the bad channel file done')

# Construct the REAL useful and all bipolar lead channel dicts.
label_dict = OrderedDict()
all_label_dict = OrderedDict()
count = 0
for i, (key, value) in enumerate(onset_ch_to_time.items()):
    result = re.match(r"([a-zA-Z]+'?)([0-9]+)", key).groups()
    new_key = key + '-' + result[0] + str(int(result[1]) + 1)
    # count is the index of remain channels; value[0] is the index of all channels
    if new_key not in global_del_channel:
        label_dict[new_key] = [count, value[0]]
        count += 1
    all_label_dict[new_key] = [i, value[0]]
print('The remaining channel dict:\n', label_dict)
print('The total channel dict:\n', all_label_dict)
if not args.check:
    with open(os.path.join(root_path, 'minus_label_dict.pkl'), 'wb') as f:
        pickle.dump(label_dict, f)
    with open(os.path.join(root_path, 'all_minus_label_dict.pkl'), 'wb') as f:
        pickle.dump(all_label_dict, f)
    print('Saving the channel dicts file done')

# Obtain the base/minimal sample rate of all files in one patient.
base_sample_rate = sample_rate_dict[min(sample_rate_dict, key=lambda x: sample_rate_dict[x])]
sample_rate_dict['base'] = base_sample_rate
print('Base sample rate is: ', base_sample_rate)
if not args.check:
    with open(os.path.join(root_path, 'sample_rate_dict.pkl'), 'wb') as f:
        pickle.dump(sample_rate_dict, f)
        print('Saving the sample rate dict file done')

# Print all the error logs.
for key, value in error_logs.items():
    if len(value) > 0:
        print('-'*10, key, 'file error logs', '-'*10)
        value = list(set(value))
        for log in value:
            print(log)
print('-' * 10, 'ALL Done', '-' * 10)
