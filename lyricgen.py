import tensorflow as tf

import numpy as np
import os
import time
import pandas as pd
import pathlib

filepath = str(pathlib.Path(__file__).parent.absolute()) + '\\'


# RNN
def make_rnn(artist):
    s = pd.read_csv(filepath + 'songs_clean.csv')
    lyrics = s.loc[s['artists'].isin([artist])]['lyrics'].values
    text = ""
    for song in lyrics:
        text += song + "\n"
    print('Length of text: {} characters'.format(len(text)))


make_rnn("Post Malone")
