# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:38:58 2023

@author: mik16
"""
#%%
import numpy as np
import pandas as pd
import mne
import os

#%%
# look all the mne montages available
# builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
# for montage_name, montage_description in builtin_montages:
#     print(f'{montage_name}: {montage_description}')
#     montage = mne.channels.make_standard_montage(montage_name)
#     fig = montage.plot()

data_path = "insert here path"

sfreq = 125
n_channels = 16
ch_types = ['eeg'] * 16
#original channel names
# ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','T4','T5','P3','Pz','P4','T6','O1','O2']
# i nomi dei canali che sto usando sono leggermente diversi da quelli veri perchè devono
#corrispondere a quelli del template ma non credo sia importarte l'effettivo nome
ch_names = ['P3','T8','P7','Pz','P4','P8','O1','O2','Fp1','Fp2','F3','Fz','F4','F8','T7','F7']
# ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','T8','P7','P3','Pz','P4','P8','O1','O2']
#add information
info = mne.create_info(n_channels, sfreq = sfreq)
info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)
#builds the selected template
montage = mne.channels.make_standard_montage('easycap-M1')
#fig = montage.plot(sphere = 0.078)
#adds the template to info
info.set_montage('easycap-M1')

i = 1
for file in os.listdir(data_path):
    filename = os.path.join(data_path, file)
    filename = filename.replace(os.sep, '/')
    
    data = pd.read_csv(filename, sep = ',', header = 4)
    data = data.drop("Sample Index", axis = 1)
        
    data = data.iloc[625:22500, 0:16]
    
    raw = mne.io.RawArray(data.transpose(), info)
    
    
    filetosave = filename[:52] + str(i) + ".set"
    i+=1
    mne.export.export_raw(filetosave, raw, fmt='eeglab', overwrite=True)




#import data and crop to desidered length

# data = pd.read_csv("D:/OneDrive - Università degli Studi di Bari/Scuola/Università/Magistrale/Health Technologies/Final Project/Data_serious/Andrea/OpenBCI-RAW-2023-01-18_17-26-44.txt", sep = ",", header = 4)
# data = data.drop("Sample Index", axis = 1)
# data = data.iloc[625:15001, 0:16]

# #set info parameters

# #creates the file with data and info
# raw = mne.io.RawArray(data.transpose(), info)
# fig = raw.plot_sensors(show_names=True)
# #export the file in .set
# mne.export.export_raw('D:/OneDrive - Università degli Studi di Bari/Scuola/Università/Magistrale/Health Technologies/Final Project/Data/test1.set', raw, fmt='eeglab', overwrite=True)


# #open .set file and extract channels data
# prova = mne.io.read_raw_eeglab('D:/OneDrive - Università degli Studi di Bari/Scuola/Università/Magistrale/Health Technologies/Final Project/Data/test_clean.set')
# df = pd.DataFrame(prova[0:16][0])
