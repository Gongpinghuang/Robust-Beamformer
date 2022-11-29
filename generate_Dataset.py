import numpy as np
from torch.utils.data import Dataset
import json
import random as rnd
import librosa as lr
from used_function import test_rir_spectrum


def generate_Dataset(speechs_meta, rirs_meta, file_json, data_meta):

    """Generate meta file of Dataset for training.
    Parameters
    ----------
    speechs_meta : meta file
        the meta file that save the information (like path, offset...) of speeches from the trainset of TIMIT
    rirs_meta : meta file
        the meta file that save the information (like path, microphones) of generated RIRs
    file_json : json file
        save the setup parameter for training speeches (like duration)
    ----------
    data_meta : data file
        the meta file to save the information (like path of RIRs, selected speeches... etc) of generated dataset
    """

    # open the meta files of training speeches and RIRs and load the content

    with open(speechs_meta) as f1:
        speech_elements = f1.read().splitlines()

    with open(rirs_meta) as f2:
        rir_elements = f2.read().splitlines()

    with open(file_json, 'r') as f3:
        params = json.load(f3)

    duration = params['duration']

    global test_result_x
    num_rirs = len(rir_elements)
    fs = 16000

    # iterate over the all generated RIRs and randomly select the speech for each rir from TIMIT train set

    for i in range(0, num_rirs):

        rir_info = json.loads(rir_elements[i])
        mics = rir_info['mics']
        rir_path = rir_info['path']
        sources = rir_info['sources']

        nMics = len(mics)

        hs, _ = lr.core.load(path=rir_path, sr=16000, mono=False)

        # randomly select the speech from train set as target source
        x_speech = json.loads(rnd.choice(speech_elements))
        x_speech_path = x_speech['path']
        x_duration = lr.core.get_duration(filename=x_speech_path)

        # test if the duration selected speech which is as target source satisfy the requirement
        if x_duration >= duration:
            x_offset = round((x_duration - duration) * rnd.uniform(0.0, 1.0) * 100) / 100
        else:
            x_offset = 0.0
        # save the information of selected speech as target source
        x_speech_json = {'duration': duration, 'offset': x_offset, 'path': x_speech_path}

        # load the target source speech
        x, _ = lr.core.load(path=x_speech_path, sr=16000, mono=True, offset=x_offset, duration=duration)

        # randomly select the speech from train set as interference source
        v_speech = json.loads(rnd.choice(speech_elements))
        v_speech_path = v_speech['path']
        v_duration = lr.core.get_duration(filename=v_speech_path)

        # test if the duration selected speech which is as target source satisfy the requirement
        if v_duration >= duration:
            v_offset = round((v_duration - duration) * rnd.uniform(0.0, 1.0) * 100) / 100
        else:
            v_offset = 0.0

        # save the information of selected speech as target source
        v_speech_json = {'duration': duration, 'offset': v_offset, 'path': v_speech_path}

        # load the target source speech
        v, _ = lr.core.load(path=v_speech_path, sr=16000, mono=True, offset=v_offset, duration=duration)

        for iMic in range(0, nMics):

            h_x = np.squeeze(hs[iMic, :])
            x = np.convolve(x, h_x, mode='same')

            try:
                # test if the result of convolution of RIR with speech is correct
                test_result_x = test_rir_spectrum(x, h_x, mics[iMic], sources[0], fs, 340)

            except test_result_x:
                print("Error: decomposition of rir is wrong")

        # save the information of generated dataset element and write them into the meta file.
        meta = {}
        meta['RIR'] = json.loads(rir_elements[i])
        meta['Speech'] = []
        meta['Speech'].append(x_speech_json)
        meta['Speech'].append(v_speech_json)
        meta_str = json.dumps(meta)
        print(meta_str)
        with open(data_meta, mode='a') as f:
            f.write(meta_str)
            f.write('\n')


if __name__ == "__main__":
    speech_meta = './meta/TIMIT_train.meta'
    rirs_meta = './meta/rirs5.meta'
    file_json = './json/speech.json'
    data_meta = './meta/train_Dataset.meta'

    generate_Dataset(speechs_meta=speech_meta, rirs_meta=rirs_meta, file_json=file_json, data_meta=data_meta)
