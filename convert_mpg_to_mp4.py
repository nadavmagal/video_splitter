import os

from moviepy.editor import VideoFileClip
import subprocess
import cv2


def convert_video_moviepy(input_file, output_file):
    clip = VideoFileClip(input_file)
    # clip = clip.resize(height=frame_height, width = frame_width)
    clip.write_videofile(output_file, codec='libx264', audio_codec='aac')


def convert_video_ffmpg(input_file, output_file):
    command = f"ffmpeg -i {input_file} -vcodec h264 -acodec mp2 {output_file}"
    subprocess.call(command, shell=True)

input_dir = r'/Users/nadav/Downloads/family_videos'
output_dir = os.path.join(input_dir, 'mp4_videos')
os.makedirs(output_dir, exist_ok=True)

all_videos = [cur_name for cur_name in os.listdir(input_dir) if cur_name.endswith('.mpg')]

for cur_name in all_videos:
    input_file = os.path.join(input_dir, cur_name)
    output_file = os.path.join(output_dir, cur_name.replace('.mpg','.mp4'))  # replace with your output file path

    try:
        convert_video_moviepy(input_file, output_file)
    except:
        print(f'fails {cur_name}')
        continue
