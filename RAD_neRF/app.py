from flask import Flask, request
from flask_cors import CORS

import subprocess
import wave
import io
import requests
app = Flask(__name__)
CORS(app, origins=['https://express-call.onrender.com/',
     'http://localhost:8081/'])
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

# # print(f'[INFO] use audio file: {Aud}')

@app.route('/get-blob-data', methods=['POST'])
def get_blob_data():
    print(f"went into get-blob-data")
    print(request)

    blob_data = request.data
    print(f"blob_data==> {blob_data}")
    # # do something with the blob data here
    # with wave.open(io.BytesIO(data), "rb") as f:
    #     # Step 3: Use the write method to write the wave file data to a new .wav file
    #     with wave.open("recording.wav", "wb") as out:
    #         out.setparams(f.getparams())
    #         out.writeframes(f.readframes(f.getnframes()))

    # RunInference = subprocess.run(['test.py', '-O', '--torso', '--pose', 'preTrained/marco.json', '--data_range', Pose_start, Pose_end, '--ckpt', 'preTrained/marco_eo.pth', '--aud', 'OI THE AUDIO THINGY FILE.npy', '--bg_img', 'data/bg.jpg', '--workspace', 'trial'])
    return 'Success'


@app.route('/')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(port=5000)
