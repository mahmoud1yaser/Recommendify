import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import Pandas to use DataFrames


df_cluster = pd.read_csv('data/cluster_recommend_dataset.csv')


def track_exist_fast(x):
    x1 = str.lower(x)
    if x1 in list(df_cluster["track_identifier"]):
        return True
    else:
        return False


def recommend_me_by_cluster(x1, n=5):
    x = str.lower(x1)
    if x in list(df_cluster["track_identifier"]):
        c_df = df_cluster[df_cluster["track_identifier"] == x][["cluster", "popularity"]]
        c_df.sort_values("popularity", ascending=False, inplace=True)
        c_no = int(c_df["cluster"][0:1])
        r_df = df_cluster[df_cluster["cluster"] == c_no][["track_uri", "artist_uri", "album_uri", "track_name", "artist_name", "album_name", "popularity"]]
        r_df.sort_values("popularity", ascending=False, inplace=True)
        final_df = r_df[["track_uri", "artist_uri", "album_uri", "track_name", "artist_name", "album_name"]][1:n + 1]
        names_id = list(final_df.iloc[:, 0])
        artist_name_id = list(final_df.iloc[:, 1])
        album_name_id = list(final_df.iloc[:, 2])
        names = list(final_df.iloc[:, 3])
        artist_name = list(final_df.iloc[:, 4])
        album_name = list(final_df.iloc[:, 5])
        return names_id, artist_name_id, album_name_id, names, artist_name, album_name
    else:
        return "Our database has no track with this name"


df_content = pd.read_csv('data/content_recommend_dataset.csv')
# We selected the most popular 10000 songs only to make our algo work faster
df_mini = df_content.sort_values("popularity", ascending=False)[:10000]

# Create CountVectorizer object to transform text into vector
track_vectorizer = CountVectorizer()

# Fit the vectorizer on "metadata" field of df_mini DataFrame
track_vectorizer.fit(df_mini['metadata'])


def track_exist_accurate(x):
    x1 = str.lower(x)
    if x1 in list(df_mini["track_identifier"]):
        return True
    else:
        return False


# Function to recommend more songs based on given song name
def recommend_me_by_content(song_name, n=5):
    song_identifier = str.lower(song_name)
    tracks = df_mini.copy()
    try:
        # Numeric columns (audio features) in track DataFrame
        num_cols = ['duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo', 'mode_no', 'mode_yes', 'speech_no',
                    'speech_yes', 'happy_no', 'happy_yes', 'popularity']
        # Create vector from "metadata" field (text data) for given song
        text_vec1 = track_vectorizer.transform(
            tracks[tracks['track_identifier'] == str(song_identifier)]['metadata']).toarray()

        # Create vector from numerical columns for given song
        num_vec1 = tracks[tracks['track_identifier'] == str(song_identifier)][num_cols].to_numpy()

        # Initialise empty list to store similarity scores
        sim_scores = []

        # For every song/track in tracks, determine cosine similarity with given song
        for index, row in tracks.iterrows():
            name = row['track_name']

            # Create vector from "metadata" field for other songs
            text_vec2 = track_vectorizer.transform(tracks[tracks['track_name'] == name]['metadata']).toarray()

            # Create vector from numerical columns for other songs
            num_vec2 = tracks[tracks['track_name'] == name][num_cols].to_numpy()

            # Calculate cosine similarity using text vectors
            text_sim = cosine_similarity(text_vec1, text_vec2)[0][0]

            # Calculate cosine similarity using numerical vectors
            num_sim = cosine_similarity(num_vec1, num_vec2)[0][0]

            # Take average of both similarity scores and add to list of similarity scores
            sim = (text_sim + num_sim) / 2
            sim_scores.append(sim)

        # Add new column containing similarity scores to tracks DataFrame
        tracks['similarity'] = sim_scores

        # Sort DataFrame based on "similarity" column
        tracks.sort_values(by=['similarity', 'popularity'], ascending=[False, False], inplace=True)
        # Create DataFrame "recommended_songs" containing 5 songs that are most similar to the given song and return
        # this DataFrame
        recommended_songs = tracks[["track_uri", "artist_uri", "album_uri", 'track_name', 'artist_name', 'album_name']][1:(1+n)]
        names_id = list(recommended_songs.iloc[:, 0])
        artist_name_id = list(recommended_songs.iloc[:, 1])
        album_name_id = list(recommended_songs.iloc[:, 2])
        names = list(recommended_songs.iloc[:, 3])
        artist_name = list(recommended_songs.iloc[:, 4])
        album_name = list(recommended_songs.iloc[:, 5])
        return names_id, artist_name_id, album_name_id, names, artist_name, album_name
    except:
        # If given song is not found in song library then display message
        print('{} not found in songs library.'.format(song_name))
