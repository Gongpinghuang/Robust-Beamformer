import librosa as lr
import numpy as np
import scipy.linalg as la
import math
import torch


def estimate_steering_vector(mics, s_position, paras):

    d_s = []
    speed = paras['speed']
    fs = paras['sample_frequency']
    frameSize = paras['frame_Size']

    nMic = mics.shape[0]

    src = np.array([s_position[0], s_position[1]])
    src -= np.mean(np.array(mics), axis=0)
    src /= np.sqrt(np.sum(src ** 2))

    for i in range(0, nMic):
        mic1_pos = mics[0] - np.mean(np.array(mics), axis=0)
        mic2_pos = mics[i] - np.mean(np.array(mics), axis=0)
        tdoa = (fs / speed) * np.dot(mic1_pos, src) - (fs / speed) * np.dot(mic2_pos, src)
        f = np.expand_dims(np.arange(0, frameSize / 2 + 1), axis=0)
        A = np.exp(-1j * 2 * np.pi * tdoa * f / frameSize)
        d_s.append(A)

    d_s = np.float32(np.concatenate(d_s, axis=0))

    return d_s


def beamformer(Ys, mics, ds, s_position, paras):

    speed = paras['speed']
    fs = paras['sample_frequency']
    frameSize = paras['frame_Size']
    r_factor = 0.01

    F = Ys.shape[2]
    M = Ys.shape[0]
    T = Ys.shape[1]
    Rd_elemets = np.ones((M, M))
    identy_M = np.identity(M)
    h_sd = np.zeros((M, F), dtype=np.complex64)

    src = np.array([s_position[0], s_position[1]])
    src -= np.mean(np.array(mics), axis=0)
    src /= np.sqrt(np.sum(src ** 2))

    for i in range(0, M):
        for j in range(i + 1, M):
            mic1_pos = mics[i] - np.mean(np.array(mics), axis=0)
            mic2_pos = mics[j] - np.mean(np.array(mics), axis=0)
            tdoa = (fs / speed) * np.dot(mic1_pos, src) - (fs / speed) * np.dot(mic2_pos, src)
            Rd_elemets[i, j] = tdoa

    Rd_elemets = np.dot(np.transpose(Rd_elemets), Rd_elemets)

    for f in range(0, F):
        Rd_f = np.sinc(2 * np.pi * f / frameSize * Rd_elemets)

        h_sd_up = np.linalg.inv((Rd_f + r_factor * identy_M) * ds[:, f])

        h_sd_blow1 = np.matmul(np.transpose(np.conj(ds[:, f])), h_sd_up)
        h_sd_blow2 = np.matmul(h_sd_blow1, ds[:, f])

        h_sd[:, f] = h_sd_up / h_sd_blow2

    h_sd = np.repeat(np.expand_dims(h_sd, 1), T, 1)

    Z = np.sum(np.conj(h_sd) * Ys, 0)

    return Z
