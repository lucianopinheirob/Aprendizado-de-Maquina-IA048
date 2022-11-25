# shows audio analysis for the given track
#export SPOTIPY_CLIENT_ID=cdf39b5c3fe34a8ca659e21f96100b9f
#export SPOTIPY_CLIENT_SECRET=497310228ce341bea13bff8c213f08f7
#export SPOTIPY_REDIRECT_URI=https://localhost:8888/callback

from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import numpy as np

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#Rap 5
#Metal 4 
#Punk 3
#Samba 2
#Reggae 1
#Eletronica 0


generos =  {'spotify:playlist:37i9dQZF1DX186v583rmzp': [0, 0, 0, 0, 0, 1],
            'spotify:playlist:37i9dQZF1DWWOaP4H0w5b0': [0, 0, 0, 0, 1, 0],
            'spotify:playlist:37i9dQZF1DX3LDIBRoaCDQ': [0, 0, 0, 1, 0, 0],
            'spotify:playlist:37i9dQZF1DWX4CmhTadwuL': [0, 0, 1, 0, 0, 0],
            'spotify:playlist:37i9dQZF1DXbSbnqxMTGx9': [0, 1, 0, 0, 0, 0],
            'spotify:playlist:37i9dQZF1DWSYVW0BVc4a3': [1, 0, 0, 0, 0, 0],
            'spotify:playlist:37i9dQZF1DX2oc5aN4UDfD': [0, 1, 0, 0, 0, 0],
            'spotify:playlist:37i9dQZF1DX9qNs32fujYe': [0, 0, 0, 0, 1, 0],
            'spotify:playlist:7s17jTD83GgeKajAMogIen': [0, 0, 0, 0, 0, 1],
            'spotify:playlist:5XEDGTA0v2FGh8O2Qid8BX': [0, 0, 0, 1, 0, 0],
            'spotify:playlist:6pW6wHPI8iecHS2EswJQVg': [0, 0, 1, 0, 0, 0],
            'spotify:playlist:3MnaeVYyif6bTqO2PO8wVU': [1, 0, 0, 0, 0, 0]}


def Pega_Tracks(pl_id, sp):
    
    offset = 0
    Tracks = []

    while True:
        response = sp.playlist_items(pl_id,
                                 offset=offset,
                                 fields='items.track.id,total',
                                 additional_types=['track'])
    
        for i in range(len(response['items'])):
            Tracks.append(response['items'][i]['track']['id'])

        offset = offset + len(response['items'])

        print(offset)
        
        if len(response['items']) == 0:
            break 

        
    
    return Tracks

def Pega_Atributos(Track, sp):
    Analysis = sp.audio_analysis(Track)
    Duration = [Analysis['track']['duration']]
    Loudness = [Analysis['track']['loudness']]
    Tempo = [Analysis['track']['tempo']]
    Time_Signature = [Analysis['track']['time_signature']]
    Key = [Analysis['track']['key']]
    Mean_Tatum_Duration = np.sum([Tatum['duration'] for Tatum in Analysis['tatums']]) / len(Analysis['tatums'])
    Mean_Segment_Duration = np.sum([Segment['duration'] for Segment in Analysis['segments']]) / len(Analysis['segments'])
    Mean_Bar_Duration = np.sum([Bar['duration'] for Bar in Analysis['bars']]) / len(Analysis['bars'])
    Mean_Section_Duration = np.sum([Section['duration'] for Section in Analysis['sections']]) / len(Analysis['sections'])
    Mean_Beat_Duration = np.sum([Beat['duration'] for Beat in Analysis['beats']]) / len(Analysis['beats'])
    Mean_Timbre = np.sum([Segments['timbre'] for Segments in Analysis['segments']], axis=0) / len(Analysis['segments'])
    Mean_Pitches = np.sum([Segments['pitches'] for Segments in Analysis['segments']], axis=0) / len(Analysis['segments'])

    print(Duration)

    Atributos = np.concatenate((Duration,
                            Loudness,
                            Tempo,
                            Time_Signature,
                            Key,
                            Mean_Tatum_Duration,
                            Mean_Segment_Duration,
                            Mean_Bar_Duration,
                            Mean_Section_Duration,
                            Mean_Beat_Duration,
                            Mean_Timbre,
                            Mean_Pitches), axis = None)
        
    return Atributos

Tracks = []
Labels = []
for key in generos.keys():
    New_Tracks = Pega_Tracks(key, sp)
    Tracks += New_Tracks
    for Track in New_Tracks: 
        Labels.append(generos[key])
    

Atributos = [Pega_Atributos(Track, sp) for Track in Tracks]

np.savetxt("Atributos.csv", Atributos, delimiter=",")
np.savetxt("Labels.csv", Labels, delimiter=",")
