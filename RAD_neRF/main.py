import os
print(os.environ.get('CUDA_PATH'))
# Check if .npy file exists in data folder

if os.path.exists('data/*.npy'):
    print("File exists")
