import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import cv2 
import os
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

def create_text_png(text, save_path, image_width=200, image_height=100, font_size=20):
    # Create a blank image with a white background
    image = Image.new("RGB", (image_width, image_height), "white")

    # Initialize drawing on the image
    draw = ImageDraw.Draw(image)

    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), text, font_size=font_size)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (image_width - text_width) // 2
    text_y = (image_height - text_height) // 2

    # Add the text to the image
    draw.text((text_x, text_y), text, fill="black", font_size=font_size)

    # Save the image as a PNG file
    image.save(save_path)
    print(f'Created {save_path}')

def video_info_check(video_name):
    data_path = '/root/capsule/data'
    video_paths = []
    base_paths = []
    for root, dirs, files in os.walk(data_path):
        if video_name + '.avi' in files:
            print('Video found')
            video_paths.append(os.path.join(root, video_name+'.avi'))
            base_paths.append(root)
        elif video_name + '.mp4' in files:
            print('Video found')
            video_paths.append(os.path.join(root, video_name+'.mp4'))
            base_paths.append(root)
    video_info = dict()
    video_info['video_detection_count'] = len(video_paths)
    video_info['output_dir'] = os.path.join('/root/capsule/results', video_name)
    create_text_png(f"Detected {len(video_paths)} video(s).", os.path.join(video_info['output_dir'], 'detected_videos.png'))
    # print .png as output
    if video_info['video_detection_count'] == 1:
        video_paths = video_paths[0]
        base_paths = base_paths[0]
        video_info['video_file'] = video_paths
        video_info['timestamps_file'] = os.path.join(base_paths, f"{video_name}.csv")
        video_info['timestamps_exist'] = os.path.exists(video_info['timestamps_file'])
        create_text_png(f"Video timestamps exist: {str(video_info['timestamps_exist'])}.", os.path.join(video_info['output_dir'], 'detected_videos_timestamps.png'))
    return video_info

def cal_video_temporal_qm(video_info):
    video_temporal_qm = dict()
    video_temporal_qm['IFI_cdf_quantile'] = np.arange(0, 105, 5).tolist()
    video_temporal_qm['IFI_cdf_value'] = None
    video_temporal_qm['IFI_range'] = None
    video_temporal_qm['ITI_cdf_quantile'] = np.arange(0, 105, 5).tolist()
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
            video_temporal_qm['IFI_cdf_value'] = np.percentile(IFI, video_temporal_qm['IFI_cdf_quantile']).tolist()
            video_temporal_qm['ITI_cdf_value'] = np.percentile(ITI, video_temporal_qm['ITI_cdf_quantile']).tolist()
            video_temporal_qm['IFI_range'] = [np.min(IFI), np.max(IFI)]
            video_temporal_qm['ITI_range'] = [np.min(ITI), np.max(ITI)]
            hist_joint_all = sns.jointplot(x = ITI, y = IFI, kind = 'hist', bins = 10,
                       stat='probability',marginal_ticks=True, 
                       marginal_kws=dict(bins=20, fill=True, stat='probability')).set_axis_labels(xlabel = 'Harp time (ms)', ylabel='Camera time')
            hist_joint_all.savefig(os.path.join(video_info['output_dir'], 'hist_joint_all.png'))
            mode_range  = (ITI>=video_temporal_qm['ITI_cdf_value'][1]) & (ITI<=video_temporal_qm['ITI_cdf_value'][-2]) & (IFI>=video_temporal_qm['IFI_cdf_value'][1]) & (IFI <=video_temporal_qm['IFI_cdf_value'][-2])
            mode_range  = (ITI<=video_temporal_qm['ITI_cdf_value'][-2]) & (IFI<=video_temporal_qm['IFI_cdf_value'][-2])
            hist_joint_peak = sns.jointplot(x = ITI[mode_range], y = IFI[mode_range], kind = 'hist', bins = 10,
                       stat='probability',marginal_ticks=True, 
                       marginal_kws=dict(bins=20, fill=True, stat='probability')).set_axis_labels(xlabel = 'Harp time (ms)', ylabel='Camera time (ms)')
            hist_joint_peak.savefig(os.path.join(video_info['output_dir'], 'hist_joint_peak.png'))
            hist_iti, ax = plt.subplots()
            ax.hist(ITI, bins=30)
            hist_iti.savefig(os.path.join(video_info['output_dir'], 'hist_iti.png'))
            hist_ifi, ax = plt.subplots()
            ax.hist(IFI, bins=30)
            hist_ifi.savefig(os.path.join(video_info['output_dir'], 'hist_ifi.png'))
    return video_temporal_qm


