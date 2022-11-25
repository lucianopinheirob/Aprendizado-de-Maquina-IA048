
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import numpy as np

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
Analysis = sp.audio_analysis('6m8AgjfI28ER6odzMxmHtR')

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

print(Atributos)

