import numpy as np
import random
import math
from used_function import test_distolist, angel_vecs

NumMic_range = [2, 16]
fs = 16000
margin = 1
speed = 340
reverb_time = [0.15, 0.9]
room = [7, 7, 3]


def rir_setup(NumMicS, MicArray_type, speed, fs, room, margin, reverb_time):

    """Generate setup parameters for the rir generator

    Parameters
    ----------
    NumMicS : int
        the number of microphones of microphone array
    MicArray_type : string: 'ULA' , 'NULA', 'UCA', 'Arbitrary'
        the type of microphone array geometry
    speed : float
        speed of sound
    fs: float
        Sampling frequency in Hz.
    room: array_like
        1D array of floats specifying the room dimensions :code:`(x, y, z)` in m.
    margin: float
        the minimum distance between the microphone and surface of simulator room
    reverb_time: float, optional
        Reverberation time (T_60) in seconds.
    ----------
    params : dictionary
        :parameters of rir for rir generator
    """

    # Select the number of microphones

    N_mics = 8

    # Define microphone array position in the room

    mic0_X = random.uniform(margin, room[0] - margin)
    mic0_Y = random.uniform(margin, room[1] - margin)
    mic0_Z = 1.5

    # Define microphone rotation
    Phi = np.pi * random.uniform(0, 360)/360

    # According to the type of microphone array, generate the microphone array

    if MicArray_type == 'ULA':

        mics = np.zeros((N_mics, 3), dtype=np.float64)
        distance_mics = random.uniform(0.005, 0.05)
        for i in range(0, N_mics):
            mic_x = 0
            mic_y = i * distance_mics

            # add rotation effect to the microphone
            mics[i, 0] = math.cos(Phi)*mic_x + math.sin(Phi)*mic_y
            mics[i, 1] = math.cos(Phi)*mic_y - math.sin(Phi)*mic_x
            mics[i, 2] = 0

        # add microphone array position to the microphone position
        mics[:, 0] += mic0_X
        mics[:, 1] += mic0_Y
        mics[:, 2] += mic0_Z

    elif MicArray_type == 'NULA':
        mics = np.zeros((N_mics, 3), dtype=np.float64)

        for i in range(1, N_mics):
            distance_mics = random.uniform(0.005, 0.05)
            mic_x = 0
            mic_y = mics[i - 1, 1] + distance_mics

            # add rotation effect to the microphone
            mics[i, 0] = math.cos(Phi)*mic_x + math.sin(Phi)*mic_y
            mics[i, 1] = math.cos(Phi)*mic_y - math.sin(Phi)*mic_x
            mics[i, 2] = 0

        # add microphone array position to the microphone position
        mics[:, 0] += mic0_X
        mics[:, 1] += mic0_Y
        mics[:, 2] += mic0_Z

    elif MicArray_type == 'UCA':
        mics = np.zeros((N_mics, 3), dtype=np.float64)
        radius = random.uniform(0.02, 0.05)
        angle = np.linspace(0, 2 * np.pi, N_mics)
        for i in range(0, N_mics):
            mics[i, 0] = radius * math.cos(angle[i])
            mics[i, 1] = radius * math.sin(angle[i])
            mics[i, 2] = 0

        # add microphone array position to the microphone position
        mics[:, 0] += mic0_X
        mics[:, 1] += mic0_Y
        mics[:, 2] += mic0_Z

    elif MicArray_type == 'Arbitrary':
        mics = []
        dis_range = [0.005, 0.3]
        mic_sample = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 0]
        mics.append(mic_sample)
        couter = 1
        while couter < N_mics:
            mic_x = random.uniform(0.0, 1.0)
            mic_y = random.uniform(0.0, 1.0)
            mic_z = 0
            if test_distolist([mic_x, mic_y], mics, dis_range):
                mics.append([mic_x, mic_y, mic_z])
                couter += 1
        mics = np.asarray(mics)

        # add microphone array position to the microphone position
        mics[:, 0] += mic0_X
        mics[:, 1] += mic0_Y
        mics[:, 2] += mic0_Z

    elif MicArray_type == 'pair':

        mics = np.zeros((2, 3), dtype=np.float64)
        distance_mics = random.uniform(0.005, 0.3)
        for i in range(0, N_mics):
            mic_x = 0
            mic_y = i * distance_mics

            # add rotation effect to the microphone
            mics[i, 0] = math.cos(Phi) * mic_x + math.sin(Phi) * mic_y
            mics[i, 1] = math.cos(Phi) * mic_y - math.sin(Phi) * mic_x
            mics[i, 2] = 0

        # add microphone array position to the microphone position
        mics[:, 0] += mic0_X
        mics[:, 1] += mic0_Y
        mics[:, 2] += mic0_Z

    else:

        print('error: The type of microphone array is wrong')
        return

    # Define sources position
    speakers = []
    while True:
        source_x = random.uniform(margin, room[0] - margin)
        source_y = random.uniform(margin, room[1] - margin)
        dis_SourcetoMic0 = math.sqrt(((source_x-mic0_X)**2)+((source_y-mic0_Y)**2))
        if 0.8 <= dis_SourcetoMic0 <= 3:
            break
    v_StoMic0 = np.array([source_x-mic0_X, source_y-mic0_Y])
    speakers.append([source_x, source_y, 1.5])


    while True:
        interfer_x = random.uniform(margin, room[0] - margin)
        interfer_y = random.uniform(margin, room[1] - margin)
        dis_InterftoMic0 = math.sqrt(((interfer_x-mic0_X)**2)+((interfer_y-mic0_Y)**2))
        v_ItoMic0 = np.array([interfer_x-mic0_X, interfer_y-mic0_Y])
        angle_StoI = angel_vecs(v_StoMic0, v_ItoMic0)
        if (1 <= dis_InterftoMic0 <= 5) and (angle_StoI >= 80):
            break
    speakers.append([interfer_x, interfer_y, 1.5])

    # Select the reverberation time

    reverberation_time = random.uniform(reverb_time[0], reverb_time[0])

    # store the parameters
    params = {}
    params['room'] = room
    params['MicArray_type'] = MicArray_type
    params['mics'] = mics.tolist()
    params['sources'] = speakers
    params['speed'] = speed
    params['fs'] = fs
    params['reverberation_time'] = reverberation_time

    return params
