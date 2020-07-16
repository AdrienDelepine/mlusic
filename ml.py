import string
import sys

import numpy as np
import pandas as pd
import pathlib

# nltk.download('punkt')
filepath = str(pathlib.Path(__file__).parent.absolute()) +'\\'

songs = ""
indices = None
distances = None
corpus = []


def open_file(name):
    global songs
    songs = pd.read_csv(filepath + name + '.csv')
    del songs['Unnamed: 0']


# Removes all duplicates and NaN lyrics
def remove_dupes_nan():
    file = pd.read_csv(filepath + 'songs.csv')
    del file['Unnamed: 0']
    file = file.drop_duplicates(subset='lyrics', keep="first")
    file.dropna(inplace=True)
    file.to_csv(filepath + 'songs_removedupes.csv')


def pick_artists(names):
    global songs
    songs = songs.loc[songs['artists'].isin(names)]


def create_lyrical_similarity():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk import word_tokenize
    from nltk import PorterStemmer

    song_lyrics = songs[['lyrics']].values

    global corpus
    global arr

    for lyric in song_lyrics:
        corpus.append(lyric[0])

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    def normalize(text):
        return stem_tokens(word_tokenize(text.lower().translate(remove_punctuation_map)))

    vect = TfidfVectorizer(tokenizer=normalize, min_df=1, stop_words="english")

    stemmer = PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T
    arr = pairwise_similarity.toarray()
    np.fill_diagonal(arr, np.nan)


def most_lyrically_similar(songName):
    index = songs.index[songs['title'] == songName].tolist()
    index = index[0]
    input_doc = corpus[int(index)]
    input_idx = corpus.index(input_doc)

    result_idx = np.nanargmax(arr[input_idx])
    print(songs.iloc[result_idx]['title'] + ' by ' + songs.iloc[result_idx]['artists'])
    print("Similarity: ", arr[input_idx][result_idx])
    print(corpus[result_idx][0:400])


def audio_features_NN(num_neighbors=5, features=None):
    if features is None:
        features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
                    'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
                    'time_signature', 'valence']
    from sklearn.neighbors import NearestNeighbors

    X = songs[features].to_numpy()
    global distances
    global indices
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)


def get_audio_features_NN(songName):
    index = songs.index[songs['title'] == songName].tolist()
    index = index[0]
    print(songs.iloc[index]['title'] + ' by ' + songs.iloc[index]['artists'])
    neighbors = indices[index][1:]
    print(neighbors)
    dists = distances[index][1:]
    for i in range(len(neighbors)):
        print(
            songs.iloc[neighbors[i]]['title'] + ' by ' + songs.iloc[neighbors[i]]['artists'] + '. Distance: ' + str(
                dists[i]))


def lyric_generation():  # from https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, LSTM
    from keras.utils import np_utils
    from keras.callbacks import ModelCheckpoint

    # nltk.download('stopwords')

    def tokenize(input):
        input = input.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(input)
        filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
        return " ".join(filtered)

    def create_model():
        newmodel = Sequential()
        newmodel.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        newmodel.add(Dropout(0.2))
        newmodel.add(LSTM(256, return_sequences=True))
        newmodel.add(Dropout(0.2))
        newmodel.add(LSTM(128))
        newmodel.add(Dropout(0.2))
        newmodel.add(Dense(y.shape[1], activation='softmax'))

        newmodel.compile(loss='categorical_crossentropy', optimizer='adam')
        return newmodel

    text = ""
    for lyric in songs[['lyrics']].values:
        text += lyric[0] + '\n\n'

    processed_input = tokenize(text)
    chars = sorted(list(set(processed_input)))
    char_to_num = dict((c, i) for i, c in enumerate(chars))

    input_len = len(processed_input)
    vocab_len = len(chars)

    seq_length = 100
    x_data = []
    y_data = []
    for i in range(0, input_len - seq_length, 1):
        in_seq = processed_input[i:i + seq_length]

        out_seq = processed_input[i + seq_length]

        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])

    n_patterns = len(x_data)
    print("Total Patterns:", n_patterns)
    X = np.reshape(x_data, (n_patterns, seq_length, 1))
    X = X / float(vocab_len)
    y = np_utils.to_categorical(y_data)

    filepath = "model_weights_saved.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    desired_callbacks = [checkpoint]

    model = create_model()
    model.fit(X, y, epochs=1, batch_size=512, callbacks=desired_callbacks)

    filename = "model_weights_saved.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    num_to_char = dict((i, c) for i, c in enumerate(chars))

    start = np.random.randint(0, len(x_data) - 1)
    pattern = x_data[start]
    print("Random Seed:")
    print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_len)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = num_to_char[index]
        seq_in = [num_to_char[value] for value in pattern]

        sys.stdout.write(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print(seq_in)
