""" top level run script """

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import cv2 
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Union
from aind_data_schema.core.quality_control import (QCEvaluation, QCMetric,
                                                   QCStatus, QualityControl,
                                                   Stage, Status)
from aind_data_schema_models.modalities import Modality
from video_utils import video_info_check, cal_video_temporal_qm, append_json
from PIL import Image, ImageDraw, ImageFont

status_pass = QCStatus(
    status=Status.PASS,
    evaluator="Automated",
    timestamp=datetime.utcnow().isoformat(),
)
status_fail = QCStatus(
    status=Status.FAIL,
    evaluator="Automated",
    timestamp=datetime.utcnow().isoformat(),
)
def raw_qc(video_name):
    print(f"Video_name: {video_name}")
    # Initializing qc
    qc = QualityControl(evaluations=[])
    QCMetric_list = []
    # video exists
    video_name = 'bottom_camera'
    video_info = video_info_check(video_name)
    video_detection_png = f'/{video_name}/detected_videos.png'
    # append to quality_cont
    cur_metric = QCMetric(
        name='video_count',
        value=video_info['video_detection_count'],
        status_history=[status_pass if video_info['video_detection_count']==1 else status_fail],
        description="Pass when equal to 1",
        reference=video_detection_png,  
    )
    QCMetric_list.append(cur_metric)
    if video_info['video_detection_count']==1:
        # video timestamps exists
        video_timestamps_detection_png = f'/{video_name}/detected_videos_timestamps.png'
        cur_metric = QCMetric(
            name = 'video_metadata',
            value = video_info['timestamps_exist'],
            status_history=[status_pass if video_info['timestamps_exist'] else status_fail],
            description='Pass when exist',
            reference=video_detection_png,
        )
        QCMetric_list.append(cur_metric)
        # frame counts equal

        # Inter frame interval distribution
        

    evaluation_video_detection = QCEvaluation(
    name=f"{video_name}_data_detection",
    modality=Modality.BEHAVIOR_VIDEOS,
    stage=Stage.PROCESSING,
    metrics=QCMetric_list,
    notes="Pass when both video and video metadata exist",
    )

    qc.evaluations.append(evaluation_video_detection)
    qc.write_standard_file(output_directory=os.path.join('/root/capsule/results', video_name))
    print(f'Quality control status {qc.status}')

    return video_info

if __name__ == "__main__": 
    # this needs to be from the user input
    video_name = 'bottom_camera'
    video_info = raw_qc(video_name)
    append_json(video_info, os.path.join(video_info['output_dir'], 'quality_analysis.json'))

