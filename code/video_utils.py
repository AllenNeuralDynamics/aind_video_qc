import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import cv2 
import os
from pathlib import Path
import json

def video_info_check(dir):
    base_name = os.path.splitext(os.path.basename(dir))[0]
    path_name = os.path.dirname(dir)
    video_info = dict()
    video_info['video_file'] = dir
    video_info['video_exist'] = os.path.exists(video_info['video_file'])
    video_info['timestamps_file'] = os.path.join(path_name, f"{base_name}.csv")
    video_info['timestamps_exist'] = os.path.exists(video_info['timestamps_file'])
    video_info['output_dir'] = os.path.join('/root/capsule/results', base_name)
    return video_info  

def cal_video_temporal_qm(video_info):
    video_temporal_qm = dict()
    video_temporal_qm['IFI_cdf_quantile'] = list(np.arange(0, 105, 5))
    video_temporal_qm['IFI_cdf_value'] = None
    video_temporal_qm['IFI_range'] = None
    video_temporal_qm['ITI_cdf_quantile'] = list(np.arange(0, 105, 5))
    video_temporal_qm['ITI_cdf_value'] = None
    video_temporal_qm['ITI_range'] = None
    video_temporal_qm['frame_count'] = None
    video_temporal_qm['timestampe_count'] = None
    hist_iti = None
    hist_ifi = None
    if not video_info['video_exist']:
        raise ValueError('Target video does not exist')
    else:
        video = cv2.VideoCapture(video_info['video_file'])
        video_temporal_qm['frame_count'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_info['timestamps_exist']:
            video_meta_data = pd.read_csv(video_info['timestamps_file'], header = None)
            video_temporal_qm['timestampe_count'] = len(video_meta_data)
            ITI = np.diff(video_meta_data[0]*1000)
            IFI = np.diff(video_meta_data[2]/1000000)
            video_temporal_qm['IFI_cdf_value'] = list(np.percentile(IFI, video_temporal_qm['IFI_cdf_quantile']))
            video_temporal_qm['ITI_cdf_value'] = list(np.percentile(ITI, video_temporal_qm['ITI_cdf_quantile']))
            video_temporal_qm['IFI_range'] = [np.min(IFI), np.max(IFI)]
            video_temporal_qm['ITI_range'] = [np.min(ITI), np.max(ITI)]
            hist_joint_all = sns.jointplot(x = ITI, y = IFI, kind = 'hist', bins = 10,
                       stat='probability',marginal_ticks=True, 
                       marginal_kws=dict(bins=20, fill=True, stat='probability')).set_axis_labels(xlabel = 'Harp time (ms)', ylabel='Camera time')
            mode_range  = (ITI>=video_temporal_qm['ITI_cdf_value'][1]) & (ITI<=video_temporal_qm['ITI_cdf_value'][-2]) & (IFI>=video_temporal_qm['IFI_cdf_value'][1]) & (IFI <=video_temporal_qm['IFI_cdf_value'][-2])
            mode_range  = (ITI<=video_temporal_qm['ITI_cdf_value'][-2]) & (IFI<=video_temporal_qm['IFI_cdf_value'][-2])
            hist_joint_peak = sns.jointplot(x = ITI[mode_range], y = IFI[mode_range], kind = 'hist', bins = 10,
                       stat='probability',marginal_ticks=True, 
                       marginal_kws=dict(bins=20, fill=True, stat='probability')).set_axis_labels(xlabel = 'Harp time (ms)', ylabel='Camera time (ms)')
            hist_iti, ax = plt.subplots()
            ax.hist(ITI, bins=30)
            hist_ifi, ax = plt.subplots()
            ax.hist(IFI, bins=30)
    return hist_iti, hist_ifi, hist_joint_all, hist_joint_peak, video_temporal_qm