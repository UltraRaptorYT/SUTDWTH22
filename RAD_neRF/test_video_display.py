#@title Display Video

import os
import glob
from IPython.display import HTML
from base64 import b64encode

import subprocess

def get_latest_file(path):

  print(f"path ==> {path}")
  dir_list = glob.glob(path)
  print(f"dir_list ==> {dir_list}")
  dir_list.sort(key=lambda x: os.path.getmtime(x))
  return dir_list[-1]

# path to trial folder

# Video = 'ngp_ep0059.mp4'
# print(f"Type of Video ==> {type(Video)}")
# print(f"Sameple Video ==> {Video}")
# Video_aud = Video.replace('.mp4', '_aud.mp4')
# print(f"Video_aud ==> {Video_aud}")
# # concat audio
# concat_audio_command = subprocess.run(['ffmpeg', '-y', '-i', Video, '-i', 'data/' + Aud, '-c:v', 'copy', '-c:a', 'aac', Video_aud])
# print(f"concat_audio_command ==> {concat_audio_command}")

# display
def show_video(video_path, video_width=450):
   
  video_file = open(video_path, "r+b").read()
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"

  return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")

show_video('C:\\Users\\Kaleb Nim\\Documents\\GitHub\\SUTDWTH22\\RAD-NeRF\\ngp_ep0059.mp4')