import numpy as np
import math
import matplotlib.pyplot as plt

def test_distolist(sample, mic_list, range):
    """
    test if the distance between the point and any microphone of microphone array beyond range
    :param sample: array like
    the position of point, :code:`(x, y, z)` in m.
    :param mic_list: list
    the list of microphone positions of array
    :param range: tupel
    the range of test
    :return flag: bool
    the test result: if not beyond range, the flag set to Ture, otherwise False
    """
    flag = True
    len_list = len(mic_list)
    for i, val in enumerate(mic_list):
        dis = math.sqrt(((sample[0]-val[0])**2)+((sample[1]-val[1])**2))
        if dis < range[0] or dis > range[1]:
            flag = False
            break

    return flag

def angel_vecs(vector1, vector2):
    """
    calculate the angel between two vectors

    """
    l_vector1 = np.sqrt(vector1.dot(vector1))
    l_vector2 = np.sqrt(vector2.dot(vector2))
    dot_product = vector1.dot(vector2)
    cos_ = dot_product/(l_vector1*l_vector2)
    radian = np.arccos(cos_)
    angle = radian**180/np.pi

    return angle

def test_rir(meta_str, rir_wav):
    """
    test if the rir is correctly generated
    :param meta_str: json file which store the information of microphone array
    :param rir_wav: the generated rir
    """
    nMics = rir_wav.shape[0]
    nsamples = int(rir_wav.shape[1]/2)
    mics = np.asarray(meta_str['mics'])
    S = np.asarray(meta_str['sources'])

    h = np.squeeze(rir_wav[0, :nsamples])
    h_x = np.arange(0, nsamples, 1)
    p1 = np.squeeze(mics[1:, 0])
    p2 = np.squeeze(mics[1:, 1])

    chosen_M1_x = np.squeeze(mics[0, 0])
    chosen_M1_y = np.squeeze(mics[0, 1])
    s1 = np.squeeze(S[0][0])
    s2 = np.squeeze(S[0][1])

    plt.figure(1)
    plt.subplot(211)
    plt.plot(chosen_M1_x, chosen_M1_y, 'go')
    plt.plot(p1, p2, 'bo')
    plt.plot(s1, s2, 'bo')
    # plt.plot(s1, s2, 'ro')
    plt.subplot(212)
    plt.plot(h_x, h)
    plt.show()

def test_rir_spectrum(s, h, mic, s_location, fs, c):
    """
    test if the result of convolution of RIR with speech is correct

    Parameter
    ---------
        s : array_like
        speech signal that is convolved with rir

        h:h : array_like
        rir, The room impulse response, shaped `(nsample, )`

        mic:array_like
        microphone position, 1D or 2D array of floats, specifying the :code:`(x, y, z)` coordinates of the receiver(s)

        s_location: array like
        source position, 1D array of floats specifying the :code:`(x, y, z)` coordinates of the source in m.

        fs: float
        Sampling frequency in Hz.

        c : float
        Sound velocity in m/s. Usually 340
    ---------
    Return
    test_flag: bool
        if the result of convolution of RIR with speech is correct, flag is True, otherwise, False
    ---------
    """
    dis = math.sqrt(((mic[0] - s_location[0]) ** 2) + ((mic[1] - s_location[1]) ** 2))
    delay_time = dis/c
    delay_nsamples = int(math.ceil(delay_time*fs))
    early_nsamples = int(math.ceil(0.04*fs))
    # calculate direct sound of rir
    h_d = np.concatenate((h[0:delay_nsamples], np.zeros(h.shape[0]-delay_nsamples)), axis=-1)
    # calculate early reflection sound of rir
    h_early = np.concatenate((np.concatenate((np.zeros(delay_nsamples), h[delay_nsamples:early_nsamples]), axis=-1),
                              np.zeros(h.shape[0]-early_nsamples)), axis=-1)
    # calculate later reflection sound of rir
    h_later = np.concatenate((np.zeros(early_nsamples), h[early_nsamples:]), axis=-1)
    # threshold of value is set to 1e-5
    th_v = 1e-5
    # compare the convolution of the speech and original rir with the sum of convolution of speech with component of rir
    orig_spec = np.convolve(s, h, mode='same')
    estimate_spec = np.convolve(s, h_d, mode='same')+np.convolve(s, h_early, mode='same')+np.convolve(s, h_later, mode='same')
    diff_spec = np.abs(orig_spec-estimate_spec)
    if diff_spec.all()>th_v:
        test_flag = False
    else:
        test_flag = True

    return test_flag
