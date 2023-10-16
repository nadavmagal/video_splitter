import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import datetime
from datetime import datetime
from moviepy.editor import *
import pandas as pd
from skimage.metrics import structural_similarity


DEBUG = False
NUM_OF_COND = 0

MIN_MOVIE_LEN = 3#30  # sec
MAX_MOVIE_LEN = 120  # sec

LARGE_DIFF_TH = 400
MIN_NUM_OF_MATHCES = 70
SSIM_TH = 0.7 #0.18
HIST_TH = 0.9


def find_sift_num_of_matches(cur_frame, prev_frame):
    # sift
    sift = cv2.xfeatures2d.SIFT_create()
    cur_kp, cur_des = sift.detectAndCompute(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY), None)
    prev_kp, prev_des = sift.detectAndCompute(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(cur_des, prev_des, k=2)
    except:
        return 0

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 80 and DEBUG:
        # -- Draw matches
        img_matches = np.empty(
            (max(cur_frame.shape[0], prev_frame.shape[0]), cur_frame.shape[1] + prev_frame.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(cur_frame, cur_kp, prev_frame, prev_kp, good, img_matches,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # -- Show detected matches
        cv2_imshow(img_matches)

    return len(good)


def diff_histogram(cur_frame, prev_frame):
    prev_hist = np.zeros([256, 3])
    cur_hist = np.zeros([256, 3])

    cur_hist_b = cv2.calcHist([cur_frame[:, :, 0]], [0], None, [256], [0, 256])
    cur_hist_g = cv2.calcHist([cur_frame[:, :, 1]], [0], None, [256], [0, 256])
    cur_hist_r = cv2.calcHist([cur_frame[:, :, 2]], [0], None, [256], [0, 256])

    prv_hist_b = cv2.calcHist([prev_frame[:, :, 0]], [0], None, [256], [0, 256])
    prv_hist_g = cv2.calcHist([prev_frame[:, :, 1]], [0], None, [256], [0, 256])
    prv_hist_r = cv2.calcHist([prev_frame[:, :, 2]], [0], None, [256], [0, 256])

    if DEBUG:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(cur_hist_b)
        plt.plot(cur_hist_r)
        plt.plot(cur_hist_g)
        plt.title('histogram cur image')
        plt.subplot(2, 1, 2)
        plt.plot(prv_hist_b)
        plt.plot(prv_hist_r)
        plt.plot(prv_hist_g)
        plt.title('histogram prv image')
        plt.show(block=False)

    r_comparison_val = cv2.compareHist(cur_hist_r, prv_hist_r, cv2.HISTCMP_CORREL)
    g_comparison_val = cv2.compareHist(cur_hist_g, prv_hist_g, cv2.HISTCMP_CORREL)
    b_comparison_val = cv2.compareHist(cur_hist_b, prv_hist_b, cv2.HISTCMP_CORREL)

    tot_comparison_val = np.mean([r_comparison_val, g_comparison_val, b_comparison_val])

    return tot_comparison_val


def split_video(input_video_path, output_directory_path, start_frame_to_crop=0):
    cap = cv2.VideoCapture(input_video_path)

    if cap.isOpened() is False:
        print('Error openning video stream or file')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'fps is {fps}')

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    is_color = True

    is_video_not_done = True
    serial_number = -1
    frame_number = 0

    start_time = start_frame_to_crop / fps
    end_time = start_time + 1

    good_mathches_vec = []
    diff_matches_vec = []
    time_stamp = []
    ssim = []
    hist_comp_val_vec = []
    is_cut_vec = []
    frame_number_vec = []

    MIN_NUMBER_OF_FRAMES = MIN_MOVIE_LEN * fps
    MAX_NUMBER_OF_FRAMES = MAX_MOVIE_LEN * fps

    while frame_number < start_frame_to_crop and cap.isOpened():
        ret, cur_frame = cap.read()
        frame_number += 1
        if ret is False:
            print('ret is false')
            is_video_not_done = False
            break

    while (is_video_not_done):
        serial_number += 1

        first_frame = True
        number_of_fames = 0
        while (cap.isOpened()):
            if first_frame:
                first_frame = False
                ret, cur_frame = cap.read()
                prev_frame = cur_frame.copy()
            else:
                prev_frame = cur_frame.copy()
                ret, cur_frame = cap.read()

            if ret is False:
                print('ret is false')
                is_video_not_done = False
                break

            if DEBUG:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(cur_frame)
                plt.subplot(1, 2, 2)
                plt.imshow(prev_frame)
                plt.show(block=False)
                plt.figure()

            hist_comp_val = diff_histogram(cur_frame, prev_frame)
            # num_of_mathches = find_sift_num_of_matches(cur_frame, prev_frame)
            num_of_mathches=1000
            score_ssim = structural_similarity(cur_frame, prev_frame, channel_axis=2)

            print(f'time:{frame_number / fps} - #mathces:{num_of_mathches}, ssim:{score_ssim}, hist:{hist_comp_val}')

            try:
                diff_of_match_between_two_frames = np.abs(good_mathches_vec[-1] - good_mathches_vec[-2])
            except:
                diff_of_match_between_two_frames = 0

            hist_comp_val_vec.append(hist_comp_val)
            good_mathches_vec.append(num_of_mathches)
            diff_matches_vec.append(diff_of_match_between_two_frames)
            time_stamp.append(frame_number / fps)
            frame_number_vec.append(frame_number)
            ssim.append(score_ssim)

            large_diff_flag = diff_of_match_between_two_frames > LARGE_DIFF_TH
            min_num_of_mathces_flag = num_of_mathches < MIN_NUM_OF_MATHCES
            ssim_flag = score_ssim < SSIM_TH
            max_num_frames_flag = number_of_fames > MAX_NUMBER_OF_FRAMES
            hist_comp_val_flag = hist_comp_val < HIST_TH

            if (sum([large_diff_flag, min_num_of_mathces_flag, ssim_flag, hist_comp_val_flag]) >= NUM_OF_COND \
                and number_of_fames > MIN_NUMBER_OF_FRAMES) \
                    or max_num_frames_flag:
                start_time = end_time
                end_time = frame_number / fps

                reason_for_cut = '_'
                reason_for_cut += 'large_diff_flag_' if large_diff_flag else ''
                reason_for_cut += 'min_num_of_mathces_flag_' if min_num_of_mathces_flag else ''
                reason_for_cut += 'ssim_flag_' if ssim_flag else ''
                reason_for_cut += 'max_num_frames_flag_' if max_num_frames_flag else ''
                reason_for_cut += 'hist_comp_val_flag_' if hist_comp_val_flag else ''

                is_cut_vec.append(reason_for_cut)
                break
            is_cut_vec.append('')
            number_of_fames += 1
            frame_number += 1

        video_len_minuts = np.floor((end_time - start_time) / 60)
        video_len_seconds = np.round(np.mod(end_time - start_time, 60))

        out_full_path = os.path.join(output_directory_path,
                                     f'se_{serial_number} len_{video_len_minuts}.{video_len_seconds}_frames:{round(start_time * fps)}_to_{round(end_time * fps)}_reason:{reason_for_cut}.mp4')
        video = VideoFileClip(input_video_path).subclip(start_time, end_time)
        video.write_videofile(out_full_path, fps=fps)

        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
        ffmpeg_extract_subclip(input_video_path, start_time, end_time, targetname=out_full_path)

        match_in_frames_pd = pd.DataFrame({'time stamp': time_stamp})
        match_in_frames_pd['frame_number'] = frame_number_vec
        match_in_frames_pd['is_cut_vec'] = is_cut_vec
        match_in_frames_pd['num of mathces'] = good_mathches_vec
        match_in_frames_pd['diff_matches_vec'] = diff_matches_vec
        match_in_frames_pd['ssim score'] = ssim
        match_in_frames_pd['hist_comp_val_vec'] = hist_comp_val_vec

        match_in_frames_pd.to_csv(os.path.join(output_directory_path, 'num_of_mathces_per_frame.csv'))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # The start :]
    # input_video_dir_path = r'/Users/nadav/Downloads/family videos'
    input_video_dir_path = r'/Users/nadav/Movies/Wondershare UniConverter15/Converted'
    video_name = '1_Title_ 1.mp4'

    input_video_full_path = os.path.join(input_video_dir_path, video_name)

    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
    parameters_string = f'diff-{LARGE_DIFF_TH}_minmatch-{MIN_NUM_OF_MATHCES}_ssimth-{SSIM_TH}_hist-{HIST_TH}_cond-{NUM_OF_COND}'

    output_directory_path = os.path.join(input_video_dir_path, f'{video_name[0:-4]}_{dt_string}_{parameters_string}')

    if not os.path.isdir(output_directory_path):
        os.makedirs(output_directory_path)
    start_frame = 0
    split_video(input_video_full_path, output_directory_path, start_frame)