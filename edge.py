from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
def split_into_scenes(video_path, output_folder):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    scenes = []
    start_time = 0

    output_file = 'output_video_with_audio.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    for i in range(1, int(duration)):
        frame = clip.get_frame(i)
        if (frame.mean() < 0.05):  # if the frame is almost black
            end_time = i - 1  # end of the scene (1 second before black frame)
            scenes.append((start_time, end_time))
            start_time = i + 1  # start of next scene (1 second after black frame)

    # append last scene
    scenes.append((start_time, int(duration)))

    for i, (start_time, end_time) in enumerate(scenes):
        output_path = f"{output_folder}/scene_{i+1}.mp4"
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)



path_to_your_video = '/Users/nadav/Downloads/family videos/1_Title_ 1.mpg'
output_folder ='/Users/nadav/Downloads/family videos/results'
split_into_scenes(path_to_your_video, output_folder)
