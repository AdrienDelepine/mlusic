import os
import time

import spotipy
from google.auth.transport import requests
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import pandas as pd
import re
import pathlib

genius = lyricsgenius.Genius(os.environ['GENIUS_CLIENT_ACCESS_TOKEN'])
genius.verbose = False
genius.remove_section_headers = True

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
filepath = str(pathlib.Path(__file__).parent.absolute()) + '\\'

acquiredAlbums = []


# Only gets songs from albums, 20 albums max, doesn't deal with duplicates, lyrics often messed up


def search(inp, stype):
    result = spotify.search(q=inp, type=stype, limit=1)[stype + 's']['items']
    if result:
        return result[0]
    print('Invalid input')
    return None


def searchForArtist(artistInput):
    item = search(artistInput, 'artist')
    if item is None:
        return None, None
    artistURI = item['uri']
    artistName = item['name']
    return artistName, artistURI


# Gets all of an artist's songs (from albums), from list of artists
def allArtistSongs(artistsInput):
    data = []
    for artistInput in artistsInput:
        artistName, artistURI = searchForArtist(artistInput)
        if artistURI is None:
            return None

        artistAlbums = spotify.artist_albums(artistURI, album_type='album', limit=50, country='US')['items']
        # artistSingles = spotify.artist_albums(artist, album_type='single')['items']
        trackIDs, albumTitles, albIDs, release_dates, trackNames = [], [], [], [], []

        numAlbs = 0
        for album in artistAlbums:
            if (not album['name'] in albumTitles) and (not album['id'] in acquiredAlbums) and (numAlbs < 20):
                albIDs.append(album['id'])
                numAlbs += 1
            albumTitles.append(album['name'])

        if not albIDs:
            print('artist has no albums or all songs are already added')
            return None
        albums = spotify.albums(albIDs)['albums']
        albIDs = []

        for album in albums:
            tracks = album['tracks']['items']
            for track in tracks:
                trackID = track['id']
                if (trackID not in trackIDs) and ("Movie Trailer" not in track['name']) \
                        and ("(Live)" not in track['name']):
                    trackIDs.append(trackID)
                    albIDs.append(album['id'])
                    release_dates.append(album['release_date'])

        b = True
        rd = 0
        while b:
            lessThan51tracks = trackIDs
            if len(trackIDs) > 50:
                lessThan51tracks = trackIDs[0:50]
                trackIDs = trackIDs[50:]
            else:
                b = False
            trueTracks = spotify.tracks(lessThan51tracks)['tracks']
            audioFeatures = spotify.audio_features(lessThan51tracks)

            for i in range(len(trueTracks)):
                trueTrack = trueTracks[i]
                audioFeature = audioFeatures[i]
                songTitle = trueTrack['name'].replace('Bonus Track', '').replace('Single Version', '').replace(
                    'Album Version', '')
                songTitle = re.sub("[\(\[].*?[\)\]]", "", songTitle)
                time.sleep(1)

                try:
                    genius_song = genius.search_song(songTitle, artistName)
                except requests.exceptions.ReadTimeout:
                    genius_song = None

                if genius_song is not None:
                    genius_song = genius_song.lyrics
                    if len(re.findall(r'\w+', genius_song)) > 1000:
                        genius_song = None

                if audioFeature is None:
                    audioFeature = {'danceability': None, 'energy': None, 'key': None, 'loudness': None, 'mode': None,
                                    'speechiness': None, 'acousticness': None, 'instrumentalness': None, 'liveness':
                                        None, 'valence': None, 'tempo': None, 'duration_ms': None,
                                    'time_signature': None}
                print(trueTrack['name'])
                data.append([trueTrack['name'], artistName, trueTrack['album']['name'], albIDs[rd], release_dates[rd],
                             trueTrack['duration_ms'], trueTrack['id'], trueTrack['popularity'], trueTrack['explicit'],
                             audioFeature['acousticness'], audioFeature['danceability'], audioFeature['energy'],
                             audioFeature['instrumentalness'], audioFeature['key'], audioFeature['liveness'],
                             audioFeature['loudness'], audioFeature['mode'], audioFeature['speechiness'],
                             audioFeature['tempo'], audioFeature['time_signature'], audioFeature['valence'],
                             genius_song])
                rd += 1
    return data


# Adds songs from requested artist to songs.csv
def yipyip(artists):
    global acquiredAlbums
    artistList = artists.split(',')

    # df = pd.DataFrame(columns=['title', 'artists', 'album', 'albumID', 'release_date', 'duration', 'spotifyID',
    #                            'popularity', 'isExplicit', 'acousticness', 'danceability', 'energy',
    #                            'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
    #                            'time_signature', 'valence', 'lyrics'])
    df = pd.read_csv(filepath + 'songs.csv')
    del df['Unnamed: 0']

    acquiredAlbums = df['albumID'].unique()
    datas = allArtistSongs(artistList)
    if datas is None:
        return None
    j = 0
    for i in range(len(df), len(df) + len(datas)):
        data = datas[j]
        j += 1
        if not data[5] in df.spotifyID.values:
            df.loc[i] = data
    df.to_csv(filepath + 'songs.csv')

    # Remove songs with duplicate lyrics or n/a fields and save to songs_clean.csv
    df = df.drop_duplicates(subset='lyrics', keep="first")
    df.dropna(inplace=True)
    df.to_csv(filepath + 'songs_clean.csv')


stoppers = ['quit', 'q']
query = input('comma seperated list of artists =>')

while not any(stopper in query for stopper in stoppers):
    yipyip(query)
    query = input('comma seperated list of artists =>')

print('done')
