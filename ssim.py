import os
import cv2
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from skimage.metrics import structural_similarity

# Define the path to the long video file
long_video_path = '/Users/nadav/Downloads/family videos/1_Title_ 1.mpg'

# Define the directory where you want to save the individual video clips
output_directory = '/Users/nadav/Downloads/family videos/results'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Open the video file
cap = cv2.VideoCapture(long_video_path)

# Initialize variables for scene change detection
prev_frame = None
scene_start_frame = 0
scene_number = 1

# Define the threshold for SSIM (adjust as needed)
ssim_threshold = 0.85
frames_counter = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if prev_frame is not None:
        frames_counter += 1
        # Convert frames to grayscale for SSIM calculation
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the SSIM between the current frame and the previous frame
        # ssim = cv2.compareStructuralSimilarity(gray_frame, prev_gray_frame)
        ssim = structural_similarity(gray_frame, prev_gray_frame, full=False)


        if ssim < ssim_threshold and frames_counter > 20:
            # Detected a scene change
            scene_end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Extract the subclip
            start_time = scene_start_frame / cap.get(cv2.CAP_PROP_FPS)
            end_time = scene_end_frame / cap.get(cv2.CAP_PROP_FPS)

            output_filename = os.path.join(output_directory, f'scene_{scene_number}.mp4')
            ffmpeg_extract_subclip(long_video_path, start_time, end_time, targetname=output_filename)

            scene_start_frame = scene_end_frame
            scene_number += 1
            frames_counter = 0

    prev_frame = frame

cap.release()
cv2.destroyAllWindows()

print("Video split into scenes successfully.")
