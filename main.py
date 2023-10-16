import os
import scenedetect
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Define the path to the long video file
long_video_path = '/Users/nadav/Downloads/family videos/1_Title_ 1.mpg'

# Define the directory where you want to save the individual video clips
output_directory = '/Users/nadav/Downloads/family videos/results'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Create a video manager and scene manager
video_manager = VideoManager([long_video_path])
scene_manager = SceneManager()

# Add a content detector to the scene manager
scene_manager.add_detector(ContentDetector())

# Perform scene detection
video_manager.set_downscale_factor()
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)
video_manager.release()

# Iterate through the detected scenes and extract subclips
for scene in scene_manager.get_scene_list():
    start_time = scene[0].get_timecode().get_seconds()
    end_time = scene[1].get_timecode().get_seconds()

    output_filename = os.path.join(output_directory, f'scene_{scene[0].get_number()}.mp4')
    ffmpeg_extract_subclip(long_video_path, start_time, end_time, targetname=output_filename)

print("Video split into scenes successfully.")
