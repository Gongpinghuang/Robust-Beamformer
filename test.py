import numpy as np
from used_function import test_rir
import librosa as lr
import json

folder_wav = './RIRs/'
extension = '.wav'
wave_pre = '{0:>05d}'.format(0)
path = folder_wav+'UCA' + wave_pre + extension
# print(path)
path_meta = './Meta/rirs.meta'
path = "farfieldpair/aadncdjpya.wav"

hs, _ = lr.core.load(path=path, sr=16000, mono=False)

print(hs.shape)
with open(path_meta) as f:
	elements = f.read().splitlines()

meta = json.loads(elements[2])
test_rir(meta, hs)

