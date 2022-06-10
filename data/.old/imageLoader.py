import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
#Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id='e2b7e92cf8684577a314a8804b97337a', client_secret='a847df678a5145d0a62381b255e4e4fd')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
metal = 'https://open.spotify.com/playlist/27gN69ebwiJRtXEboL12Ih?si=a1144c027d6f4a4c'

def getPlaylist_albums(link):
    playlist_URI = link.split("/")[-1].split("?")[0]
    album_uris = set()
    popularity = []
    offs = 0
    while True:
        track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI,offset=offs)["items"]]
        if track_uris == []:
            break
        for t in track_uris:
            t_info = sp.track(t)
            album_uris.add(t_info['album']['uri'])
            popularity.append(t_info['popularity'])
        offs += 100
    return album_uris, sum(popularity)/len(popularity)


infos = getPlaylist_albums('https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF?si=2659db7399f0496d')
c1 = "https://open.spotify.com/playlist/1h0CEZCm6IbFTbxThn6Xcs?si=e1c141c4afea4a0e"
c2 = 'https://open.spotify.com/playlist/37i9dQZF1DWWEJlAGA9gs0?si=ede098e0e39f4ac8'
#infos_classical = getPlaylist_albums(c1)[0].union(getPlaylist_albums(c2)[0])
                              
def getImages(albums):
    i=1
    for uri in albums:
        local_file = open(f'album_pics\Top\local_pic{i}.png','wb')
        try:
            image_url = sp.album(uri)['images'][1]['url']
            resp = requests.get(image_url, stream=True)
            local_file.write(resp.content)
            local_file.close()
            i+=1
        except:
            print(f'Invalid Pic Detected : {uri}')
    return
#print('\n',infos)
getImages(infos[0])