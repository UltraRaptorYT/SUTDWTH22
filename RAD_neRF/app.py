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

def resetDatafolder():
    """
    Function to delete files used in previous post requrest
    """
    files = os.listdir('data')
    for file in files:
        if file.endswith('.wav') or file.endswith('.npy') or file.endswith('.mp4'):
            os.remove(os.path.join('data', file))

@app.route('/get-blob-data', methods=['POST'])
def get_blob_data():
    print(f"went into get-blob-data")
    try:
        resetDatafolder()
        print(f"deleted files in data folder")
    except:
        print(f"no files in data folder or error")

    # get audio blob from frontend
    data = request.files
    print(f"data['audioBlob']==>", data['audioBlob'])
    data['audioBlob'].save('data/nvp_HY.wav')


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
    """
    Takes in .mp4 file of face video (without audio) and .wav file of audio and
    Returns the video with audio concatenated denoted by '_aud' suffix
    """
    try:
        concat_audio_command = subprocess.run(['ffmpeg', '-y', '-i', Video, '-i', 'data/nvp.wav', '-c:v', 'copy', '-c:a', 'aac', Video_aud], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f'Concat audio failed:')
        print(e.stderr.decode())

    # Convert video to base64
    video_file = open(Video_aud, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"

    return video_url

# -------------------------------------------------------------------------------TESTING ROUTES BELOW THIS LINE--------------------------------------------------------------------------------
#Test audio processing
@app.route("/test_audio_process",methods=['POST'])
def test_audio_processing():
    """
    Test Audio-spatial Decomposition from .wav file from frontend 
    """
    print(f"went into test-audio-processing")
    try:
        run_extract = subprocess.run(['python', 'nerf/asr.py', '--wav', 'data/nvp_HY.wav', '--save_feats'], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f'Extract audio features failed:')
        print(e.stderr.decode())

    return "success"

@app.route("/test_audio_process_default",methods=['POST'])
def test_audio_processing_default():
    """
    Test Audio-spatial Decomposition from .wav file from frontend 
    """
    print(f"went into test_audio_processing_default")
    try:
        run_extract = subprocess.run(['python', 'nerf/asr.py', '--wav', 'data/nvp.wav', '--save_feats'], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f'Extract audio features failed:')
        print(e.stderr.decode())

    return "success"

@app.route('/test-return-video-data', methods=['POST'])
def get_audio_data():
    """
    Test .mp4 to base64 conversion and return to frontend
    """
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

    # Convert video to base64
    video_file = open(Video, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"

    # Save video url to file
    with open('TESTvideo_url.txt', 'w') as f:
        f.write(video_url)
    return video_url 

@app.route('/test-concat-audio-video', methods=['POST'])
def test_concat_audio_video():
    
    Video = 'trail/results/test_video_noAudio.mp4' # Hard coded for now
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

    # Concat audio with video
    """
    Takes in .mp4 file of face video (without audio) and .wav file of audio and
    Returns the video with audio concatenated denoted by '_aud' suffix
    """
    try:
        concat_audio_command = subprocess.run(['ffmpeg', '-y', '-i', Video, '-i', 'data/nvp.wav', '-c:v', 'copy', '-c:a', 'aac', Video_aud], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f'Concat audio failed:')
        print(e.stderr.decode())
    
    return 'Concat audio success'


@app.route('/')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(port=5000)
