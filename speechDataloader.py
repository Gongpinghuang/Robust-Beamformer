import numpy as np
from torch.utils.data import Dataset
import json
import random as rnd
import librosa as lr
import math

class SpeechData(Dataset):

    """ Dataset loader

    Parameters
    ----------
    data_meta : meta file
        the meta file that save the information (like path, offset...) of training dataset

    file_json : json file
        save the setup parameter for training data (like frame size, hop size)
    ----------
    Return
    ----------
    Ys : numpy array with size M x T x F (M: microphone number, T: number of frame, F: number of frequency bin)
       power spectrums of received signal for each microphone

    Micpos: numpy array with M x 2
       the normalized microphone positions as labels of training
    """

    def __init__(self, data_meta, file_json):

        with open(data_meta) as f1:
            self.elements = f1.read().splitlines()

        with open(file_json, 'r') as f2:
            features = json.load(f2)

        self.frameSize = features['frame_Size']
        self.hopSize = features['hop_Size']
        self.epsilon = features['epsilon']

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):

        fs = 16000
        audio = json.loads(self.elements[item])
        mics = audio['RIR']['mics']
        rir_path = audio['RIR']['path']
        sources = audio['RIR']['sources']
        room = audio['RIR']['room']

        nMics = len(mics)
        nSours = len(sources)

        # load rir signal
        hs, _ = lr.core.load(path=rir_path, sr=16000, mono=False)

        duration = audio['Speech'][0]['duration']
        N = round(duration * fs)

        # load the target and interference speech
        x_path = audio['Speech'][0]['path']
        x_offset = audio['Speech'][0]['offset']

        v_path = audio['Speech'][1]['path']
        v_offset = audio['Speech'][1]['offset']

        x_speech, _ = lr.core.load(path=x_path, sr=16000, mono=True, offset=x_offset, duration=duration)
        if x_speech.shape[0] <= 16000:
            x_speech = np.concatenate((x_speech, np.zeros(16000 - x_speech.shape[0])), axis=-1)

        v_speech, _ = lr.core.load(path=v_path, sr=16000, mono=True, offset=v_offset, duration=duration)
        if v_speech.shape[0] <= 16000:
            v_speech = np.concatenate((v_speech, np.zeros(16000 - v_speech.shape[0])), axis=-1)

        Ys = []
        MicPos = []
        mean_mic = np.sum(mics, axis=0) / len(mics)

        # convolve the speech signal with the rir for each microphone
        for iMic in range(0, nMics):

            # convolve the target speech with rir for target source
            h_x = np.squeeze(hs[iMic, :])
            x = np.convolve(x_speech, h_x, mode='same')

            # adjust the target speech according to SNR
            snr = rnd.uniform(0, 5.0)
            x /= np.sqrt(np.mean(x ** 2))
            x *= 10.0 ** (snr / 20.0)

            # convolve the interference speech with rir for interference source
            h_v = np.squeeze(hs[iMic, :])
            v = np.convolve(v_speech, h_v, mode='same')

            # generate the white noise with 15-40 dB
            wgn = math.sqrt((rnd.uniform(15, 40))) * np.random.randn(1, int(duration*fs))
            wgn = np.squeeze(wgn)

            # obtain the received signal of microphone
            v = v+wgn
            y = x+v

            # obtain the log power spectrum of received signal
            y_sift = lr.core.stft(y, n_fft=self.frameSize, hop_length=self.hopSize)
            y_sift = np.log(np.abs(y_sift) ** 2 + self.epsilon) - np.log(self.epsilon)
            y_sift = np.expand_dims(y_sift, axis=0)
            Ys.append(y_sift)

            MicPo = np.array(mics[iMic]) - mean_mic

            # normolize the microphone position
            MicPo = np.array([MicPo[0] / room[0], MicPo[1] / room[1]])
            MicPos.append(MicPo)


        Ys = np.float32(np.concatenate(Ys))
        MicPos = np.float32(np.concatenate(MicPos, axis=0))

        return Ys, MicPos
