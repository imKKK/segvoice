import pickle
import numpy as np

from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture


def read_wav(wav):
    return wavfile.read(wav)


def get_pos_feat(sig, fs):
    return mfcc(sig, fs)


def get_feat(wav):
    fs, sig = read_wav(wav)
    mfcc_feature = mfcc(sig, fs)
    return mfcc_feature


def fit(src, mixture=32):
    gmm = GaussianMixture(mixture)
    if type(src) == str:
        src = get_feat(src)
    gmm.fit(src)
    return gmm


def score(gmm, feat):
    return gmm.score(feat)


def save_model(wav, path):
    gmm = fit(wav)
    with open(path, 'wb') as f:
        pickle.dump(gmm, f)


def predict(path, wav):
    with open(path, 'rb') as f:
        gmm = pickle.load(f)
    feat = get_feat(wav)
    return score(gmm, feat)


