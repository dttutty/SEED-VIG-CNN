import scipy.io as scio
import os
import pickle
import numpy as np


eeg_folders = {\
    'EEG_Feature_2Hz' : './SEED-VIG/EEG_Feature_2Hz/',
    'EEG_Feature_5Bands' : './SEED-VIG/EEG_Feature_5Bands/',
    'Forehead_EEG_Feature_2Hz': './SEED-VIG/Forehead_EEG/EEG_Feature_2Hz/', 
    'Forehead_EEG_Feature_5Bands': './SEED-VIG/Forehead_EEG/EEG_Feature_5Bands/',}
eeg_sub_features = ['psd_movingAve', 'psd_LDS', 'de_movingAve', 'de_LDS']

eog_folder = './SEED-VIG/EOG_Feature/'
eog_sub_features = ['features_table_ica', 'features_table_minus', 'features_table_icav_minh']

perclos_folder = './SEED-VIG/perclos_labels/'

inputs = {}

filenames = os.listdir(perclos_folder)

for filename in filenames:
    # 合并23个人的eeg信息
    for eeg_feature, eeg_folder in eeg_folders.items():
        eeg_mat = scio.loadmat(os.path.join(eeg_folder, filename))
        for eeg_sub_feature in eeg_sub_features:
            new_list = eeg_mat[eeg_sub_feature].transpose(1,0,2)
            concat_feature_name = eeg_feature+'_'+eeg_sub_feature
            inputs[concat_feature_name] = np.concatenate((inputs[concat_feature_name], new_list), axis=0) if concat_feature_name in inputs else new_list
    
    # 合并23个人的eog信息
    eog_mat = scio.loadmat(os.path.join(eog_folder, filename))
    for eog_sub_feature in eog_sub_features:
        new_list = eog_mat[eog_sub_feature]
        concat_feature_name = 'EOG_Feature'+'_'+eog_sub_feature
        inputs[concat_feature_name] = np.concatenate((inputs[concat_feature_name], new_list), axis=0) if concat_feature_name in inputs else new_list
    
    # 合并23个人的parclos信息
    parclos_mat = scio.loadmat(os.path.join(perclos_folder, filename))
    new_list = parclos_mat['perclos']
    try:
        outputs
    except NameError:
        outputs = new_list
    else:
        outputs = np.concatenate((outputs, new_list), axis=0)
   
with open('inputs.pickle', 'wb') as handle:
    pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

with open('outputs.pickle', 'wb') as handle:
    pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
