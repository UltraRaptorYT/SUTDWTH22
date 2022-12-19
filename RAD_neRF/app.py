from flask import Flask, request
from flask_cors import CORS
import os
import subprocess
import wave
import io
import requests

from base64 import b64encode
app = Flask(__name__)
CORS(app)
# @markdown ####**Settings:**
Person = 'engm'  # @param ['obama', 'marco', 'engm', 'chris']
Audio = 'custom'  # @param ['intro', 'nvp', 'custom']
Background = 'default'  # @param ['default', 'custom']
Pose_start = 0  # @param {type: 'integer'}
Pose_end = 100  # @param {type: 'integer'}


# #@title Extract audio features
# if Audio == 'custom':
#     # need to wait to download the ASR model
#     run_extract = subprocess.run(['python', 'nerf/asr.py', '--wav', 'data/' + Aud, '--save_feats'])
#     %run nerf/asr.py --wav data/{Aud} --save_feats

# # print(f'[INFO] use audio file: {Aud}')'


@app.route('/get-blob-data', methods=['POST'])
def get_blob_data():
    print(f"went into get-blob-data")
    data = request.files
    print(f"data['audioBlob']==>", data['audioBlob'])
    data['audioBlob'].save('data/nvp_HY.wav')


    # # do something with the blob data here

    # Code to extract audio features
    """
    Takes in .wav file, performs Audio-spatial Decomposition and returns .npy file of audio features
    """
    try:
        run_extract = subprocess.run(['python', 'nerf/asr.py', '--wav', 'data/nvp.wav', '--save_feats'], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f'Extract audio features failed:')
        print(e.stderr.decode())


    # # # Code to run inference CUDA broke this AAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # """
    # Takes in .npy file of audio features, runs inference and returns .mp4 file of face video (without audio)
    # """
    # try:
    #     runInference = subprocess.run(['python', 'test.py', '-O', '--torso', '--pose', 'data/pose.json', '--data_range', '0', '100', '--ckpt', 'pretrained/model.pth', '--aud', 'data/nvp_eo.npy', '--bg_img', 'data/bg.jpg'], check=True, stderr=subprocess.PIPE)
    # except subprocess.CalledProcessError as e:
    #     print(f'Run inference failed:')
    #     print(e.stderr.decode())
    # print(f"runInference==> {runInference}")
    
    
    Video = 'trail/results/ngp_ep0059.mp4' # Hard coded for now
    Video_aud = Video.replace('.mp4', '_aud.mp4')

    # Concat audio with video
    """Returns"""
    try:
        concat_audio_command = subprocess.run(['ffmpeg', '-y', '-i', Video, '-i', 'data/nvp.wav', '-c:v', 'copy', '-c:a', 'aac', Video_aud], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f'Concat audio failed:')
        print(e.stderr.decode())

    # Convert video to base64
    video_file = open(Video_aud, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"

    return video_url

@app.route('/test-return-video-data', methods=['POST'])
def get_audio_data():
    print(f"went into get_audio_data")

    Video = 'trial/results/ngp_ep0059_aud.mp4' # Hard coded for now
    Video_aud = Video.replace('.mp4', '_aud.mp4')
    if os.path.exists(Video_aud):
        print('Video_aud exists.')
    else:
        print("==>> Video_aud: ", Video_aud)
        print('Video_aud does not exist.')

    if os.path.exists(Video):
        print('Video exists.')
    else:
        print("==>> Video: ", Video)
        print('Video does not exist.')

    # # Concat audio with video
    # """Returns"""
    # try:
    #     concat_audio_command = subprocess.run(['ffmpeg', '-y', '-i', Video, '-i', 'data/nvp.wav', '-c:v', 'copy', '-c:a', 'aac', Video_aud], check=True, stderr=subprocess.PIPE)
    # except subprocess.CalledProcessError as e:
    #     print(f'------------------------------------Concat audio failed:-------------------------------------------------')
    #     print(e.stderr.decode())

    # Convert video to base64
    video_file = open(Video, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"

    # Save video url to file
    with open('TESTvideo_url.txt', 'w') as f:
        f.write(video_url)
    return video_url 


@app.route('/')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(port=5000)
