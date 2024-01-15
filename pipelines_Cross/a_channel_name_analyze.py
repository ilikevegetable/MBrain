import argparse
import glob
import os
import pickle
import re
import shutil
import sys

import numpy as np
import pandas as pd
from mne.io import read_raw_edf


parser = argparse.ArgumentParser(description='ChannelProcess')
parser.add_argument('--file_dir', type=str, default='/data/eeggroup/new_data2/02GJX/',
                    help='Should give an absolute path including all .edf files and label files of one patient.')
parser.add_argument('--overwrite', action='store_true',
                    help='The flag of whether to overwrite the channel name files if they have existed.')
parser.add_argument('--electrode_name', nargs='*', type=str, default=["Ci", "SMA", "mFCi", "iFCi"],
                    help='Should give a list including all the electrode names displayed in the software. '
                         'It is Case Sensitive and should not include any space.')
parser.add_argument('--electrode_num', nargs='*', type=int, default=[12, 16, 12, 16],
                    help='Should give the number of leads corresponding to the electrode_name list.')
argv = sys.argv[1:]
args = parser.parse_args(argv)
"""
Patients' electrode information:
02GJX
electrode_name = ["Ci", "SMA", "mFCi", "iFCi"]
electrode_num = [12, 16, 12, 16]
01TGX
electrode_name = ["A\'", "H\'", "PH\'", "OB\'", "FI\'", "CI\'", "TI\'", "PI\'", "SM\'", "MF\'"]
electrode_num = [16, 12, 16, 16, 12, 12, 12, 16, 12, 12]
05ZLH
electrode_name = ["A", "B", "C", "D", "E", "F", "G", "H"]
electrode_num = [16, 16, 16, 12, 16, 16, 16, 16]
06ZYJ
electrode_name = ["A\'", "H\'", "FI\'", "OB\'", "FC\'", "TI\'", "ACC\'", "PI\'"]
electrode_num = [16, 16, 12, 16, 16, 12, 16, 16]
"""


print('-'*10, 'Start moving all files to a single file folder', '-'*10)
file_dir = args.file_dir
file_name_list = []
for file in os.listdir(file_dir):
    if file.split('.')[-1] == 'edf' or file.split('.')[-1] == 'csv':
        if not os.path.exists(file_dir + file) or os.path.exists(file_dir + file.split('.')[0] + '.mat'):
            continue
        print(file)
        file_name = file.split('.')[0]
        src_file_list = glob.glob(file_dir + file_name + '.*')
        print(src_file_list)

        dst_path = file_dir + file_name + '/'
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for src_file in src_file_list:
            _, f_name = os.path.split(src_file)
            shutil.move(src_file, dst_path + f_name)
print('-'*10, 'Done', '-'*10)

file_name_list = []
for file in os.listdir(file_dir):
    file_path = os.path.join(file_dir, file)
    if os.path.isdir(file_path):
        file_name_list.append(file_path + '/' + file + '.edf')
print('Total number of files:', len(file_name_list))
print('All file names:\n', file_name_list)


print('-'*10, 'Start saving the electrode information', '-'*10)
electrode_name = args.electrode_name
electrode_num = args.electrode_num
print('electrode_name', electrode_name)
print('electrode_num', electrode_num)
with open(file_dir + 'electrode_information.pkl', 'wb') as f:
    pickle.dump([electrode_name, electrode_num], f)
print('-'*10, 'Done', '-'*10)


print('-'*10, 'Start analyzing the channel names of every file', '-'*10)
for file_name in file_name_list:
    print('Processing the file:', file_name)
    save_file = '/'.join(file_name.split('/')[:-1]) + '/' + file_name.split('/')[-1].split('.')[0] + '_channelname.csv'
    if os.path.exists(save_file) and not args.overwrite:
        continue

    raw_data = read_raw_edf(file_name, preload=False)
    ch_names = raw_data.info['ch_names']
    print('The number of recording channels in this file:', len(ch_names))

    name_to_index = [-1 for _ in range(np.sum(electrode_num))]
    ch_names_tight = []
    for ch_index in range(len(ch_names)):
        ch = ch_names[ch_index]
        ch_tight = ch[4:].replace(' ', '')
        ch_names_tight.append(ch_tight)

        elec_name = re.match(r"[A-Za-z]+'?", ch_tight)
        if elec_name is not None:
            elec_name = elec_name.group()
            index = 0
            count = 0
            while index < len(electrode_name) and elec_name != electrode_name[index]:
                count += electrode_num[index]
                index += 1
            if index < len(electrode_name):
                ch_name = re.match(elec_name + r"[0-9]+", ch_tight)
                if ch_name is not None:
                    ch_name = ch_name.group()
                    ch_names_tight[-1] = ch_name
                    elec_num = int(ch_name.split(elec_name)[-1])
                    if name_to_index[count + elec_num - 1] == -1:
                        name_to_index[count + elec_num - 1] = ch_index

    print('The index of all valid channel names in the complete name list:\n', name_to_index)
    channel_name_list = []
    origin_channel_name = []
    for re_index in name_to_index:
        channel_name_list.append([ch_names_tight[re_index], re_index])
        origin_channel_name.append(ch_names[re_index])
    print('The list of all valid channel names in the original name list (with space):\n', origin_channel_name)
    channel_name_list = pd.DataFrame(channel_name_list, columns=['channelName', 'index'])
    print('The final valid channel names and corresponding indexes:\n', channel_name_list)
    invalid_name_list = channel_name_list[channel_name_list['index'].isin([-1])]['channelName'].tolist()
    if len(invalid_name_list) > 0:
        raise ValueError('Some channels are not included:\n', invalid_name_list)
    channel_name_list.to_csv(save_file, index=False)
    print('Processing file', file_name, 'done')
print('-'*10, 'Done', '-'*10)
