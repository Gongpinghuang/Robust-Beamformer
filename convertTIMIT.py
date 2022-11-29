import numpy as np
import os
from sphfile import SPHFile
import glob
import json


path = './TIMIT/TIMIT/database/TIMIT/TIMIT/TEST'
meta_path = './Meta/TIMIT_test.meta'

files = os.listdir(path)

for file in files:
    state_path = os.path.join(path, file)
    if os.path.isdir(state_path):
        speaker_names = os.listdir(state_path)
        for name in speaker_names:
            name_path = os.path.join(state_path, name)
            sph_path = name_path + '/*.WAV'
            sph_files = glob.glob(sph_path)
            for i in sph_files:
                meta = {}
                sph = SPHFile(i)
                i = i.replace(".WAV", "_n.wav")
                print(i)
                sph.write_wav(filename=i)
                meta['path'] = i
                meta_str = json.dumps(meta)
                with open(meta_path, mode='a') as f:
                    f.write(meta_str)
                    f.write('\n')


print("Completed")

