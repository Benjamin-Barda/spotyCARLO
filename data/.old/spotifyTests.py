import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import matplotlib.pyplot as plt

#Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id='e2b7e92cf8684577a314a8804b97337a', client_secret='a847df678a5145d0a62381b255e4e4fd')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
classical = "https://open.spotify.com/playlist/1h0CEZCm6IbFTbxThn6Xcs?si=e1c141c4afea4a0e"
classical_2 = 'https://open.spotify.com/playlist/37i9dQZF1DWWEJlAGA9gs0?si=ede098e0e39f4ac8'
metal = 'https://open.spotify.com/playlist/27gN69ebwiJRtXEboL12Ih?si=a1144c027d6f4a4c'

def getPlaylist_data(link):
    playlist_URI = link.split("/")[-1].split("?")[0]
    features = []
    offs = 0
    while True:
        track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI,offset=offs)["items"]]
        if track_uris == []:
            break
        features += sp.audio_features(track_uris)
        offs += 100
    return pd.DataFrame(features).drop(['liveness','id','uri','track_href','analysis_url','type'], axis=1)

classic_dataset = pd.concat([getPlaylist_data(classical),getPlaylist_data(classical_2)])
metal_dataset = getPlaylist_data(metal)
df_All = pd.concat([classic_dataset,metal_dataset])

data = df_All[['energy','loudness','acousticness']]
scaler = StandardScaler().fit(data)
standard_data = scaler.transform(data)



