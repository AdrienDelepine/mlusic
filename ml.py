import string
import sys

import numpy as np
import pandas as pd
import pathlib

# nltk.download('punkt')

filepath = str(pathlib.Path(__file__).parent.absolute()) + '\\'

songs = None
indices = None
distances = None
corpus = []


def open_file(name):
    global songs
    songs = pd.read_csv(filepath + name + '.csv')
    del songs['Unnamed: 0']


def get_songs():
    return songs


def pick_rows(cols, rows):
    filtered = songs
    for i in range(len(cols)):
        filtered = filtered.loc[songs[cols[i]].isin([rows[i]])]
    return filtered


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

    if type(index) is list:
        if not index:
            raise ValueError()
        index = index[0]
    input_doc = corpus[int(index)]
    input_idx = corpus.index(input_doc)

    result_idx = np.nanargmax(arr[input_idx])

    # print(songs.iloc[result_idx]['title'] + ' by ' + songs.iloc[result_idx]['artists'])
    # print("Similarity: ", arr[input_idx][result_idx])
    # print(corpus[result_idx][0:400])
    return {'artist': songs.iloc[result_idx]['artists'], 'song': songs.iloc[result_idx]['title'] + ' by ' + songs.iloc[result_idx]['artists'],
            "similarity": arr[input_idx][result_idx], "lyrics": corpus[result_idx][0:800]}


def create_audio_features_NN(num_neighbors=5, features=None):
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
    print(songName)
    index = songs.index[songs['title'] == songName].tolist()
    if type(index) is list:
        if not index:
            raise ValueError()
        index = index[0]

    print(songs.iloc[index]['title'] + ' by ' + songs.iloc[index]['artists'])
    neighbors = indices[index][1:]
    print(neighbors)
    dists = distances[index][1:]
    for i in range(len(neighbors)):
        print(
            songs.iloc[neighbors[i]]['title'] + ' by ' + songs.iloc[neighbors[i]]['artists'] + '. Distance: ' + str(
                dists[i]))


def lyricInfo():
    from nltk import word_tokenize
    from nltk import PorterStemmer
    from nltk.corpus import stopwords

    lyrics = songs.lyrics.values
    lyricsJoined = ""
    for lyric in lyrics:
        lyricsJoined += lyric + " | "

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    def normalize(tokens):
        return stem_tokens(tokens)

    # Stems words and finds unique words, total words, and average words per song
    stemmer = PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    lyricTokens = word_tokenize(lyricsJoined.lower().translate(remove_punctuation_map))
    stemmedLyrics = normalize(lyricTokens)
    stemLyricsSet = set(stemmedLyrics)
    uniqueWordCount = len(stemLyricsSet)
    print("unique words", uniqueWordCount)
    totalNumWords = len(stemmedLyrics)
    print("total words", totalNumWords)
    avgWordsPerSongs = totalNumWords / len(songs)
    print("average words per song", avgWordsPerSongs)

    from collections import Counter
    allWordCount = Counter(stemmedLyrics)
    print("all words count", allWordCount)
    # removing stop words
    all_stopwords = stopwords.words('english')
    all_stopwords.extend(['yeah', 'dont', 'aint', 'cant', 'im', 'got', 'get', "’", "go", "like", "right", "know", "oh"])
    lyrics_without_sw = [word for word in lyricTokens if not word in all_stopwords]
    lyrics_without_sw_set = set(lyrics_without_sw)
    sw_removed_wordcount = Counter(lyrics_without_sw)
    print("no sw wordcount", sw_removed_wordcount)

open_file('songs_clean')
create_audio_features_NN()
create_lyrical_similarity()


# while True:
#     call = input("\nopen file_name, clean, artist artist_names, lyricsim song_name, audiosim song_name, lyricinfo =>")
#     args = call.split(' ')
#     fun = args[0]
#     if fun == 'open':
#         open_file(args[1])
#     elif fun == 'artist':
#         print([call[7:]])
#         pick_rows('artists', [call[7:]])
#     elif fun == 'lyricsim':
#         try:
#             most_lyrically_similar(call[9:])
#         except ValueError:
#             print("Invalid track")
#     elif fun == 'audiosim':
#         try:
#             get_audio_features_NN(call[9:])
#         except ValueError:
#             print("Invalid track")
#     elif fun == 'lyricinfo':
#         lyricInfo()
