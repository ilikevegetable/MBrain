import os
import pickle
import sys
import argparse
import scipy
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy import io


parser = argparse.ArgumentParser(description='BrainRegionProcess')
parser.add_argument('--load_dir', type=str, default='/data/eeggroup/new_data2/02GJX/',
                    help='Should give an absolute path including .mat and label dict files of one patient.')
parser.add_argument('--read_csv', action='store_true',
                    help='The flag of whether to directly read the csv file rather than the mat file because the '
                         'mat file may have some errors. One can correct the csv file manually and read it.')
argv = sys.argv[1:]
args = parser.parse_args(argv)


root_path = args.load_dir
print('Brain region process path:', root_path)
with open(os.path.join(root_path, 'minus_label_dict.pkl'), 'rb') as f:
    label_dict = pickle.load(f)
with open(os.path.join(root_path, 'all_minus_label_dict.pkl'), 'rb') as f:
    all_label_dict = pickle.load(f)
print('The remaining channel dict:\n', label_dict)
print('The total channel dict:\n', all_label_dict)

if not args.read_csv:
    # Process the .mat brain region file (may change)
    mat_file = None
    for file in os.listdir(root_path):
        if file[-3:] == 'mat':
            mat_file = file
            break
    print('Find the brain region file:', mat_file)

    # Transform the original data type to easy version
    features_struct = scipy.io.loadmat(os.path.join(root_path, mat_file))
    features = features_struct['LEAD_COORDINATE']
    channel_names = features[0][0][2].reshape(-1)
    channel_names = np.array([x for x in channel_names]).flatten()
    brain_regions = features[0][0][1].reshape(-1)
    df = pd.DataFrame([channel_names, brain_regions]).T

    save_path = os.path.join(root_path, mat_file.split('.')[0] + '.csv')
    df.to_csv(save_path, index=False)
else:
    # Process the .csv brain region file
    csv_file = None
    for file in os.listdir(root_path):
        if file[-3:] == 'csv':
            csv_file = file
            break
    print('Find the brain region file:', csv_file)
    save_path = os.path.join(root_path, csv_file)

# Read the easy version of the brain region information
brain_info = pd.read_csv(save_path)
print('The brain region data frame is:\n', brain_info)
# Remove the space of channel names
brain_info['0'] = brain_info['0'].apply(lambda x: x.replace(' ', ''))
brain_info_dict = dict(zip(brain_info['0'], brain_info['1']))
print('The brain region dict is:\n', brain_info_dict)

# Construct the dict only including the remaining channels
brain_dict = OrderedDict()
for ch, index in label_dict.items():
    ch_names = ch.split('-')[0]
    brain_index = brain_info_dict[ch_names]
    # Do not delete the channels in 0 brain region forcefully because the doctor may label them
    ch_list = brain_dict.setdefault(brain_index, [])
    ch_list.append(index[0])
print('The remaining channel to brain region dict:\n', brain_dict)
with open(os.path.join(root_path, 'brain_dict.pkl'), 'wb') as f:
    pickle.dump(brain_dict, f)

# Construct the dict including all the channels
channel_dict = OrderedDict()
for ch, index in all_label_dict.items():
    ch_names = ch.split('-')[0]
    brain_index = brain_info_dict[ch_names]
    ch_list = channel_dict.setdefault(brain_index, [])
    ch_list.append(index[0])
print('All the channel to brain region dict:\n', channel_dict)
with open(os.path.join(root_path, 'channel_dict.pkl'), 'wb') as f:
    pickle.dump(channel_dict, f)
print('-' * 10, 'ALL Done', '-' * 10)
