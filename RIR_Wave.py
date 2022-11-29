import numpy as np
import rir_generator as rir

def rir_wave(params):
    """
    call the rir generator function to generate the rirs

    Parameter
    --------
    params: dictionary
    the dictionary that save the parameter for generating RIR

    c : float
        Sound velocity in m/s. Usually 340
    fs : float
        Sampling frequency in Hz.
    r : array_like
        1D or 2D array of floats, specifying the :code:`(x, y, z)` coordinates of the receiver(s)
        in m. Must be of shape :code:`(3,)` or :code:`(x, 3)` where :code:`x`
        is the number of receivers.
    s : array_like
        1D array of floats specifying the :code:`(x, y, z)` coordinates of the source in m.
    L : array_like
        1D array of floats specifying the room dimensions :code:`(x, y, z)` in m.

    --------
    Return:     h : array_like
        The room impulse response, shaped `(nsample, len(r))`

    """

    # load the parameters
    c = params['speed']
    fs = params['fs']
    r = np.asarray(params['mics'])
    L = params['room']
    S = np.asarray(params['sources'])
    reverb_time = params['reverberation_time']

    # call the rir generator function to generate the rirs for target and interference source
    h_source = rir.generate(c, fs, r, S[0], L, reverberation_time=reverb_time, nsample=8000)
    h_interference = rir.generate(c, fs, r, S[1], L, reverberation_time=reverb_time, nsample=8000)

    # stitching two rirs
    hs = np.concatenate((h_source, h_interference), axis=1)

    return hs
