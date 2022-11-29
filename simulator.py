import rir_generator as rir
import numpy as np
import json

import scipy.io.wavfile

from RIR_SetUP import rir_setup
from RIR_Wave import rir_wave
from scipy.io.wavfile import write


Num_samples = 10000
NumMic_range = [2, 16]
speed = 340
fs = 16000
reverb_time = [0.15, 0.9]
margin = 1
room = [7, 7, 3]

Num_ULA = round(Num_samples*0.1)
Num_NULA = round(Num_samples*0.1)
Num_UCA = round(Num_samples*0.1)
Num_ArbitryA = Num_samples - Num_ULA - Num_NULA - Num_UCA

folder_wav = './RIRs/'
path_json = './Meta/rirs.meta'
extension = '.wav'

for i in range(0, Num_ULA):

    params = rir_setup(NumMic_range, 'ULA', speed, fs, room, margin, reverb_time)
    wave = rir_wave(params)
    wave_pre = '{0:>05d}'.format(i)
    path_wav = folder_wav+'ULA' + wave_pre + extension
    write(path_wav, fs, wave)
    params['path'] = path_wav
    with open(path_json, mode='a') as f:
        meta_str = json.dumps(params)
        print(params)
        f.write(meta_str)
        f.write('\n')

for i in range(560, Num_NULA):

    params = rir_setup(NumMic_range, 'NULA', speed, fs, room, margin, reverb_time)
    wave = rir_wave(params)
    wave_pre = '{0:>05d}'.format(i)
    path_wav = folder_wav+'NULA' + wave_pre + extension
    write(path_wav, fs, wave)
    params['path'] = path_wav
    with open(path_json, mode='a') as f:
        meta_str = json.dumps(params)
        print(params)
        f.write(meta_str)
        f.write('\n')

#
for i in range(456, Num_UCA):

    params = rir_setup(NumMic_range, 'UCA', speed, fs, room, margin, reverb_time)
    wave = rir_wave(params)
    wave_pre = '{0:>05d}'.format(i)
    path_wav = folder_wav+'UCA' + wave_pre + extension
    write(path_wav, fs, wave)
    params['path'] = path_wav
    with open(path_json, mode='a') as f:
        meta_str = json.dumps(params)
        print(params)
        f.write(meta_str)
        f.write('\n')

for i in range(0, Num_ArbitryA):

    params = rir_setup(NumMic_range, 'Arbitrary', speed, fs, room, margin, reverb_time)
    wave = rir_wave(params)
    wave_pre = '{0:>05d}'.format(i)
    path_wav = folder_wav+'Arbitrary' + wave_pre + extension
    write(path_wav, fs, wave)
    params['path'] = path_wav
    with open(path_json, mode='a') as f:
        meta_str = json.dumps(params)
        print(params)
        f.write(meta_str)
        f.write('\n')



