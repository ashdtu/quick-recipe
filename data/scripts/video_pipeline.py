import os
import errno
from enum import Enum
import time 

import cv2
import numpy as np
import torch
from torchvision import transforms
from math import floor, ceil
from random import randint
from yt_dlp import YoutubeDL

def downld_vids(vid_file: str, dest: str):
    download_list = []
    missing_list = []
    if not os.path.exists(vid_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                vid_file)
    
    dest_path = os.path.abspath(dest)
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    with open(vid_file) as f:
        youtube_ids = f.read().strip().split('\n')
        print(youtube_ids)
        for yt_id in youtube_ids:
            status, vid_file_path = downld_vid(yt_id, dest_path)
            if status == 0:
                print(f'[INFO] Downloaded youtube video with id: {yt_id}')
                download_list.append(yt_id)
                yield yt_id, vid_file_path
            else:
                print(f'[INFO] Failed to download youtube video with id: {yt_id}')
                missing_list.append(yt_id)
    
    # save missing and download video list
    d_file = os.path.join(dest, 'download_list.txt')
    with open(d_file, 'w') as d:
        d_cnt = len(download_list)
        print(f'[INFO] Writing ({d_cnt}) downloaded video list to file.')
        d.write('\n'.join(download_list))
    
    m_file = os.path.join(dest, 'missing_file.txt')
    with open(m_file, 'w') as m:
        m_cnt = len(missing_list)
        print(f'[INFO] Writing ({m_cnt}) missing video list to file.')
        m.write('\n'.join(missing_list))
    
    print("Completed!")


def downld_vid(yt_id: str, dest:str):
    vid_url = f"https://www.youtube.com/watch?v={yt_id.strip()}"
    vid_file_path = os.path.join(dest, yt_id)
    # download the video
    #info_dict = ydl.extract_info(vid_url, download=False)
    return os.system(f"yt-dlp -o '{vid_file_path}' -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 '{vid_url}' "), f'{vid_file_path}.mp4'


###########################################################################
#- - - - - - - - - - - - - - Video Processing - - - - - - - - - - - - - - #
###########################################################################

def process_videos(vid_file: str, src: str):
    pass


def process_video(vid_path: str, dest: str, req_fps: int):
    if not os.path.exists(vid_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                vid_path)
    
    vid = cv2.VideoCapture(vid_path)
    vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
    dest_path = os.path.abspath(dest)
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    
    print(f"[INFO] Processing video file: {vid_path}, fps: {vid_fps}.")
    frame_cnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = floor(frame_cnt / vid_fps)
    gap = max(ceil(frame_cnt / dur / req_fps), 1)
    print('frame_cnt:', frame_cnt, 'duration:', dur, 'interval:', gap)
    frames = []
    seed = randint(0, gap)
    f_idx = 0
    for i in range(frame_cnt):
        is_read, frame = vid.read()
        if (i + seed) % gap == 0:
            store_frame(frame, f_idx, dest)
            f_idx += 1
    print("Total no. of frames:", f_idx)
    vid.release()
    #             if (i + 1) % fp_clip == 0:
    #                 clip_cnt += 1
    #                 clip_path = 'clip_{0:03d}'.format(clip_cnt)
    # else:
    #     print("Storing as CSV")
    #     clip_cnt = 1
    #     vid_tensor = []
    #     for i in range(frame_cnt):
    #         is_read, frame = vid.read()
    #         vid_tensor.append(convert_tensor(frame))
    #         if (i + 1) % fp_clip == 0:
    #             clip = torch.stack(vid_tensor)
    #             torch.save(clip, os.path.join(dest_path, 'clip_{0:03d}.pt'.format(clip_cnt)))
    #             vid_tensor = []
    #             clip_cnt += 1
    #             break
    #     if frame_cnt % fp_clip != 0:
    #         clip = torch.stack(vid_tensor)
    #         torch.save(clip, 'clip_{0:03d}.pt'.format(clip_cnt))
    
def store_frame(frame, f_idx, dest):
    path = os.path.join(dest, f'frame_{f_idx}.jpg')
    cv2.imwrite(path, frame)
    
    

def vid_feature_extraction():
    pass