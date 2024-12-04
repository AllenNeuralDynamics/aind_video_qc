""" top level run script """

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import cv2 
import os
from pathlib import Path
import json
from video_utils import video_info_check, cal_video_temporal_qm

if __name__ == "__main__": 
    # this needs to be from the user input
    dir = '/root/capsule/data/behavior_711042_2024-09-13_09-19-15/behavior-videos/bottom_camera.avi'
    video_info = video_info_check(dir)
    os.makedirs(video_info['output_dir'], exist_ok=True)
    json_file = os.path.join(video_info['output_dir'], 'video_qm.json')
    print(json_file)
    # will update to create if none, append if exist
    with open(os.path.join(video_info['output_dir'], "video_qm.json"), "w") as file:
        json.dump(video_info, file, indent=4, sort_keys=True)
    file.close()
    
    hist_iti, hist_ifi, hist_joint, hist_joint_peak, video_temporal_qm = cal_video_temporal_qm(video_info)

    # Read the existing data
    with open(json_file, 'r') as file:
        try:
            existing_data = json.load(file)
        except json.JSONDecodeError:
            existing_data = []  # Initialize as an empty list if file is empty

    # Check if the file is a list or dictionary and append accordingly
    if isinstance(existing_data, list):
        existing_data.append(video_temporal_qm)
    elif isinstance(existing_data, dict):
        existing_data.update(video_temporal_qm)
    else:
        print("Unexpected JSON structure")

    # Write the updated data back to the file
    with open(json_file, 'w') as file:
        json.dump(existing_data, file, indent=4, sort_keys=True)

    # save figures
    hist_ifi.savefig(os.path.join(video_info['output_dir'], 'hist_interframe-intervals.png'))
    print(f"Saved in {os.path.join(video_info['output_dir'], 'hist_interframe-intervals.png')}")
    hist_iti.savefig(os.path.join(video_info['output_dir'], 'hist_intertime-intervals.png'))
    hist_joint.savefig(os.path.join(video_info['output_dir'], 'joint_dist_interframe-intervals_intertime-intervals.png'))
    hist_joint_peak.savefig(os.path.join(video_info['output_dir'], 'joint_dist_interframe-intervals_intertime-intervals95quantile.png'))


