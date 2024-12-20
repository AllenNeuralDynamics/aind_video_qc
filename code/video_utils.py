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

def append_json(dict_name, json_dir):
    if not os.path.exists(json_dir):
        with open(json_dir, "w") as file:
            json.dump(dict_name, file, indent=4, sort_keys=True)
        file.close()
        print(f'Saved to {json_dir}.')
    else:
        with open(json_dir, 'r') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []  # Initialize as an empty list if file is empty
        file.close()
        if isinstance(existing_data, list):
            existing_data.append(dict_name)
        elif isinstance(existing_data, dict):
            existing_data.update(dict_name)
        else:
            print("Unexpected JSON structure")
        
        with open(json_dir, 'w') as file:
            json.dump(existing_data, file, indent=4, sort_keys=True)
        file.close()
        print(f'Appended to {json_dir}.'

def print_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    video_length = frame_count / frame_rate  # Total video length in seconds

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        exit()

    # Overlay the frame count and video length on the frame
    text = f"Frames: {frame_count}  Length: {video_length:.2f} sec"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green
    thickness = 2

    # Get text size for positioning
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 10  # Position from the left
    text_y = frame.shape[0] - 10  # Position from the bottom

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Display the first frame
    cv2.imshow("First Frame", frame)

    # Wait for a key press and close the display window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()

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
    os.makedirs(video_info['output_dir'], exist_ok=True)
    create_text_png(f"Detected {len(video_paths)} video(s).", os.path.join(video_info['output_dir'], 'detected_videos.png'))
    # print .png as output
    if video_info['video_detection_count'] == 1:
        video_paths = video_paths[0]
        base_paths = base_paths[0]
        video_info['video_file'] = video_paths
        video_info['timestamps_file'] = os.path.join(base_paths, f"{video_name}.csv")
        video_info['timestamps_exist'] = os.path.exists(video_info['timestamps_file'])
        create_text_png(f"Metadata exist: {str(video_info['timestamps_exist'])}.", os.path.join(video_info['output_dir'], 'detected_videos_timestamps.png'))
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
    if video_info['video_detection_count']==0:
        raise ValueError('Target video does not exist')
    elif video_info['video_detection_count']>1:
        raise ValueError('More than one video detected.')
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



def extract_frames(video_info, n_frames=1000):
    """
    Efficiently extracts frames from the video at regular intervals, storing them in a NumPy array.
    
    Args:
    - video_path (str): Path to the video file.
    - n_frames (int): Number of frames to extract.

    Returns:
    - frames (numpy array): Array containing the frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_info['video_file'])
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return None

    # Get the total number of frames in the video
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval between frames to sample
    frame_interval = total_frame_count // n_frames if total_frame_count > n_frames else 1

    # Pre-allocate space for the frames
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read first frame.")
        cap.release()
        return None
    
    # Get the frame dimensions (height, width, channels)
    height, width, channels = frame.shape
    frames = np.zeros((n_frames, height, width, channels), dtype=np.uint8)

    # Read frames at regular intervals and store them
    for i in range(n_frames):
        frame_number = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Jump to the desired frame
        ret, frame = cap.read()
        if not ret:
            break
        frames[i] = frame

    # Release the video capture object
    cap.release()

    return frames

def generate_focus_report(frames, focus_threshold=60, 
                             output_dir='focus_analysis_output'):
    """
    Analyzes each frame to determine sharpness using Laplacian variance and outputs:
    - A PDF report summarizing the analysis.
    - Example images corresponding to different sharpness percentiles.

    Args:
    - frames (list of numpy arrays): The frames to analyze.
    - focus_threshold (float): The variance threshold for focus detection.
    - output_dir (str): Root directory to save all output files (PDF, plot, example frames).

    Returns:
    - None
    """

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Construct paths for saving outputs
    output_pdf = os.path.join(output_dir, 'focus_analysis_report.pdf')
    output_plot = os.path.join(output_dir, 'focus_distribution.png')
    output_examples_dir = os.path.join(output_dir, 'focus_example_frames')
    
    # Ensure example frames directory exists
    Path(output_examples_dir).mkdir(parents=True, exist_ok=True)

    sharpness_values = []  # List to store the Laplacian variance for each frame
    focus_results = []     # List to store focus status (True or False)

    # Analyze each frame for sharpness
    for i, frame in enumerate(frames):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        focus_variance = laplacian.var()

        sharpness_values.append(focus_variance)
        is_focus = focus_variance > focus_threshold
        focus_results.append(is_focus)
    
    #  Plot sharpness distribution
    plt.figure(figsize=(8, 6))
    plt.hist(sharpness_values, bins=30, color='gray', edgecolor='black')
    plt.axvline(focus_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({focus_threshold})')
    plt.xlabel('Laplacian Variance (Sharpness)')
    plt.ylabel('Frequency')
    plt.title('Sharpness Distribution (Laplacian Variance)')
    plt.legend()
    plt.savefig(output_plot)
    plt.close()

    # Calculate focus statistics
    total_frames = len(frames)
    out_of_focus_count = len([x for x in focus_results if not x])
    out_of_focus_percentage = (out_of_focus_count / total_frames) * 100
    
    # Generate sharpness percentiles
    percentiles = [0, 10, 50, 90, 100]
    percentile_values = np.percentile(sharpness_values, percentiles)
    
    # Save example frames at each percentile
    example_frames = {}
    for percentile, value in zip(percentiles, percentile_values):
        closest_idx = np.argmin(np.abs(np.array(sharpness_values) - value))
        example_frame = frames[closest_idx].copy()
        filename = os.path.join(output_examples_dir, f'frame_{percentile}percent.png')
        title = f"{percentile}th Percentile (Sharpness: {value:.2f})"
        cv2.putText(example_frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(filename, example_frame)
        example_frames[percentile] = filename

    # Generate PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Write summary
    pdf.cell(0, 10, txt="Focus Analysis Report", ln=True, align='C')
    pdf.cell(0, 10, txt=f"Frames out of focus: {out_of_focus_count} / {total_frames} ({out_of_focus_percentage:.2f}%)")

    # Add sharpness distribution plot
    pdf.image(output_plot, x=10, y=pdf.get_y()+10, w=100)

    # Add example frames
    pdf.cell(0, 10, txt="Example Frames by Sharpness Percentile:", ln=True, align='R')
    for percentile, filename in example_frames.items():
        if pdf.get_y() > 240:  # Check if nearing the bottom of the page
            pdf.add_page()
        pdf.image(filename, x=125, y=pdf.get_y(), h=50)
        pdf.ln(50)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"Report saved as {output_pdf}")
    print(f"Example frames saved in {output_examples_dir}")

    return percentiles, percentile_values.tolist()


def generate_contrast_report(frames, contrast_threshold=1, 
                             output_dir='contrast_analysis_output'):
    """
    Analyzes each frame to determine contrast using pixel intensity variance and outputs:
    - A PDF report summarizing the analysis.
    - Example images corresponding to different contrast percentiles with titles.
    - Saves all files in the specified output directory.

    Args:
    - frames (list of numpy arrays): The frames to analyze.
    - contrast_threshold (float): The variance threshold for determining low contrast.
    - output_dir (str): Root directory to save all output files (PDF, plot, example frames).

    Returns:
    - None
    """

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Construct paths for saving outputs
    output_pdf = os.path.join(output_dir, 'contrast_analysis_report.pdf')
    output_plot = os.path.join(output_dir, 'contrast_distribution.png')
    output_examples_dir = os.path.join(output_dir, 'contrast_example_frames')
    
    # Ensure example frames directory exists
    Path(output_examples_dir).mkdir(parents=True, exist_ok=True)

    contrast_values = []  # List to store contrast (variance) values for each frame
    contrast_results = []  # List to store contrast status (low/high)

    # Analyze each frame for contrast
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # # method 1: variance of intensities
        # contrast_value = gray_frame.var()

        # optional replacement method 2: RMS contrast
        contrast_value = np.std(gray_frame) / np.mean(gray_frame)

        contrast_values.append(contrast_value)
        is_low_contrast = contrast_value < contrast_threshold
        contrast_results.append(is_low_contrast)

    # Plot contrast distribution
    plt.figure(figsize=(8, 6))
    plt.hist(contrast_values, bins=30, color='gray', edgecolor='black')
    plt.axvline(contrast_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({contrast_threshold})')
    plt.xlabel('Pixel Intensity Variance (Contrast)')
    plt.ylabel('Frequency')
    plt.title('Contrast Distribution (Pixel Intensity Variance)')
    plt.legend()
    plt.savefig(output_plot)
    plt.close()

    # Calculate contrast statistics
    total_frames = len(frames)
    low_contrast_count = len([x for x in contrast_results if x])
    low_contrast_percentage = (low_contrast_count / total_frames) * 100

    # Generate contrast percentiles
    percentiles = [0, 10, 50, 90, 100]
    percentile_values = np.percentile(contrast_values, percentiles)

    # Save example frames at each percentile
    example_frames = {}
    for percentile, value in zip(percentiles, percentile_values):
        closest_idx = np.argmin(np.abs(np.array(contrast_values) - value))
        example_frame = frames[closest_idx].copy()
        filename = os.path.join(output_examples_dir, f'frame_{percentile}percent.png')
        title = f"{percentile}th Percentile (Contrast: {value:.2f})"
        cv2.putText(example_frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(filename, example_frame)
        example_frames[percentile] = filename

    # Generate PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Write summary
    pdf.cell(0, 10, txt="Contrast Analysis Report", ln=True, align='C')
    pdf.cell(0, 10, txt=f"Frames with low contrast: {low_contrast_count} / {total_frames} ({low_contrast_percentage:.2f}%)")

    # Add contrast distribution plot
    pdf.image(output_plot, x=10, y=pdf.get_y()+10, w=100)

    # Add example frames
    pdf.cell(0, 10, txt="Example Frames by Contrast Percentile:", ln=True, align='R')
    for percentile, filename in example_frames.items():
        if pdf.get_y() > 240:  # Check if nearing the bottom of the page
            pdf.add_page()
        pdf.image(filename, x=125, y=pdf.get_y(), h=50)
        pdf.ln(50)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"Report saved as {output_pdf}")
    print(f"Example frames saved in {output_examples_dir}")
    
    return percentiles, percentile_values.tolist()


def generate_intensity_report(frames, lower_threshold=30, upper_threshold=40, 
                             output_dir='intensity_analysis_output'):

    """
    Analyzes frames for pixel intensity and outputs:
    - A PDF report summarizing the analysis.
    - Example images corresponding to different intensity percentiles with their intensity histograms.

    Args:
    - frames (list of numpy arrays): The frames to analyze.
    - lower_threshold (int): Minimum acceptable mean brightness (0-255).
    - upper_threshold (int): Maximum acceptable mean brightness (0-255).
    - output_dir (str): Root directory to save all output files (PDF, plot, example frames).

    Returns:
    - None
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Construct paths for saving outputs
    output_pdf = os.path.join(output_dir, 'intensity_analysis_report.pdf')
    output_plot = os.path.join(output_dir, 'intensity_distribution.png')
    output_examples_dir = os.path.join(output_dir, 'intensity_example_frames')
    
    # Ensure example frames directory exists
    Path(output_examples_dir).mkdir(parents=True, exist_ok=True)

    mean_intensities = []  # List to store mean intensity for each frame
    exposure_results = []  # List to store exposure classification (under/over/proper)

    # Analyze each frame for intensity
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_frame)
        mean_intensities.append(mean_brightness)
        if mean_brightness < lower_threshold:
            exposure_results.append('underexposed')
        elif mean_brightness > upper_threshold:
            exposure_results.append('overexposed')
        else:
            exposure_results.append('proper')

    # Plot overall intensity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(mean_intensities, bins=60, color='gray', edgecolor='black')
    plt.axvline(lower_threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Underexposure ({lower_threshold})')
    plt.axvline(upper_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Overexposure ({upper_threshold})')
    plt.xlabel('Mean Intensity')
    plt.ylabel('Frequency')
    plt.title('Mean Intensity over Frames')
    plt.legend()
    plt.savefig(output_plot)
    plt.close()

    # Calculate statistics
    total_frames = len(frames)
    underexposed_count = exposure_results.count('underexposed')
    overexposed_count = exposure_results.count('overexposed')
    properly_exposed_count = exposure_results.count('proper')

    underexposed_percentage = (underexposed_count / total_frames) * 100
    overexposed_percentage = (overexposed_count / total_frames) * 100
    properly_exposed_percentage = (properly_exposed_count / total_frames) * 100

    # Generate intensity percentiles
    percentiles = [0, 10, 50, 90, 100]
    percentile_values = np.percentile(mean_intensities, percentiles)

    # Save example frames and their histograms
    example_frames = {}
    for percentile, value in zip(percentiles, percentile_values):
        closest_idx = np.argmin(np.abs(np.array(mean_intensities) - value))
        example_frame = frames[closest_idx].copy()

        # Add title to the image
        title = f"{percentile}th Percentile (Intensity: {value:.2f})"
        cv2.putText(example_frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save the example frame
        frame_filename = os.path.join(output_examples_dir, f'frame_{percentile}percent.png')
        cv2.imwrite(frame_filename, example_frame)
        example_frames[percentile] = frame_filename

        # Create intensity histogram for the frame
        gray_frame = cv2.cvtColor(frames[closest_idx], cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(4, 3))
        plt.hist(gray_frame.ravel(), bins=256, range=(0, 256), color='gray', edgecolor='black')
        plt.xticks(fontsize=8)
        plt.yticks([], [])  # Remove y-axis labels
        plt.xlabel("Pixel Intensity", fontsize=8)
        histogram_filename = os.path.join(output_examples_dir, f'frame_{percentile}percent_histogram.png')
        plt.savefig(histogram_filename, bbox_inches='tight')
        plt.close()
        example_frames[f"{percentile}_histogram"] = histogram_filename

    # Generate PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Write summary
    pdf.cell(0, 10, txt="Intensity Analysis Report", ln=True, align='C')
    pdf.cell(0, 10, txt=f"Underexposed frames: {underexposed_count} / {total_frames} ({underexposed_percentage:.2f}%)", ln=True)
    pdf.cell(0, 10, txt=f"Overexposed frames: {overexposed_count} / {total_frames} ({overexposed_percentage:.2f}%)", ln=True)
    pdf.cell(0, 10, txt=f"Properly exposed frames: {properly_exposed_count} / {total_frames} ({properly_exposed_percentage:.2f}%)", ln=True)
    pdf.ln(10)

    # Add overall intensity distribution plot
    pdf.cell(0, 10, txt="Intensity Distribution:", ln=True)
    pdf.image(output_plot, x=10, y=pdf.get_y() + 10, w=180)
    pdf.add_page()

    # Add example frames and histograms
    pdf.cell(0, 10, txt="Example Frames by Intensity Percentile:", ln=True)
    for percentile in percentiles:
        if pdf.get_y() > 240:
            pdf.add_page()
        frame_file = example_frames[percentile]
        hist_file = example_frames[f"{percentile}_histogram"]
        pdf.image(frame_file, x=10, y=pdf.get_y(), h=50)
        pdf.image(hist_file, x=110, y=pdf.get_y(), h=50)
        pdf.ln(50)

    # Save the PDF
    pdf.output(output_pdf)
    print(f"Report saved as {output_pdf}")
    print(f"Example frames and histograms saved in {output_examples_dir}")

    return percentiles, percentile_values.tolist()
