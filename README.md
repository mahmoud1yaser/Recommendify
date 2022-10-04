# Spotify-Recommendation-System-Website
Recommendation Systems are used essentially in Spotify! <br>
- Spotify use different types of Recommendation Systems which are:
    - Collaborative Filtering Algorithm (Based on users interactions of different track)
    - Content Based Filtering (Based on users demographics and tracks attributes)
    - Natural Language Processing (Based on analyzing tracks lyrics) <br>
- In this project we build our Recommendation System to recommend similar songs to the user input, we used different algorithms which are:
    - Content Based Filtering
    - Clustering 
<hr>


## App video Review
[Recommendify](https://drive.google.com/file/d/1kutCUfNSivR7NXNFNVcBIK0D1lhBnkAO/view?usp=sharing)

## Understanding the Dataset
- The Spotify million playlist dataset consists of a single JSON dictionary with three fields:
   * **date** - the date the challenge set was generated. This should be "2018-01-16 08:47:28.198015"
   * **version** - the version of the challenge set. This should be "v1"
   * **playlists** - an array of 10,000 incomplete playlists. Each element in this array contains the following fields:
      * **pid** - the playlist ID
      * **name** - (optional) - the name of the playlist. For some challenge playlists, the name will be missing.
      * **num_holdouts** - the number of tracks that have been omitted from the playlist
      * **tracks** - a (possibly empty) array of tracks that are in the playlist. Each element of this array contains the following fields:
         * **pos** - the position of the track in the playlist (zero offset)
         * **track_name** - the name of the track
         * **track_uri** - the Spotify URI of the track
         * **artist_name** - the name of the primary artist of the track
         * **artist_uri** - the Spotify URI of the primary artist of the track
         * **album_name** - the name of the album that the track is on
         * **album_uri** -- the Spotify URI of the album that the track is on
         * **duration_ms** - the duration of the track in milliseconds
      * **num_samples** the number of tracks included in the playlist
      * **num_tracks** - the total number of tracks in the playlist.
- The **Audio Features** are scraped from Spotify API using spotipy library, which are:
    - **acousticness** (A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.)
    - **analysis_url** (A URL to access the full audio analysis of this track. An access token is required to access this data.)
    - **danceability** (Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.)
    - **duration_ms** (The duration of the track in milliseconds.)
    - **energy** (Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.)
    - **id** (The Spotify ID for the track.)
    - **instrumentalness** (Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.)
    - **key** (The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.)
    - **liveness** (Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.)
    - **loudness** (The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.)
    - **Mode** (indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.)
    - **speechiness** (Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.)
    - **tempo** (The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
    - **time_signature** (An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4".)
    - **track_href** (A link to the Web API endpoint providing full details of the track.)
    - **type** (The object type.)
    - **uri** (The Spotify URI for the track.)
    - **valence** (A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).)
    <hr>

## Data Scraping
We made data scraping using spotipy library and got over the data of 4000 playlists with their tracks:
```
p1, p2, p3, p4, p5 = all_uri[:20000], all_uri[2000:40000], all_uri[40000:60000], all_uri[60000:80000], all_uri[80000:]
api_features = []

for uri in tqdm(p1):
    try:
        api_features.append(sp.audio_features(uri)[0])
    except:
        pass
```

## Preprocessing and Sentiment Analysis

We got unique tracks and unique playlists in different dataframes and extracted additional features using `one hot encoding`:
- `mode_no` & `mode_yes`
    - Extracted from the `mode` audio feature
- `speech_no` &	`speech_yes`
    - Extracted from the `speechiness` audio feature
- `happy_no` & `happy_yes`
    - Extracted from the `polarity` audio feature
- `popularity`
    - Extracted from counting the playlist - `track, album, & artist` followers and giving weights to each of them to construct the popularity feature

So, they will be very helpful in understanding the track feature

Before modelling in `Recommendation System by Clustering` we scaled the data using standardization to shift the distribution to have a mean of zero and a standard deviation of one.
```
from sklearn.preprocessing import StandardScaler

# We will scale our data using standard scaler
scaler=preprocessing.StandardScaler()

# We notice that we exclude the loudness column because if if we try to standardize it, we will get many null values.
positive_numerical_columns = numerical_columns[0:4]+numerical_columns[5:]
df_scaled=pd.DataFrame(scaler.fit_transform(track[positive_numerical_columns]), columns = track[positive_numerical_columns].columns)
```

## EDA
**Univariate Analysis:**
- We extracted the artists popularity feature and got the **top 50 artists** and this algorithm worked very well:
```
df_modified["playlist_followers_artist"] = df_modified.groupby("artist_name")["num_followers"].transform("count")
df_modified["playlist_followers_artist"]

df1 = df_modified.sort_values("playlist_followers_artist", ascending = False)
df1["artist_name"].head(40000).nunique()
``` 
- We extracted the album popularity feature and got the **top 50 albums** and this algorithm worked very well:
```
df_modified["playlist_followers_album"] = df_modified.groupby("album_name")["num_followers"].transform("count")
df_modified["playlist_followers_album"]

df2 = df_modified.sort_values("playlist_followers_album", ascending = False)
df2["album_name"].head(18500).nunique()
``` 
- We extracted the track popularity feature and got the **top 50 track** and this algorithm worked very well:
```
df_modified["playlist_followers_track"] = df_modified.groupby("track_name")["num_followers"].transform("count")
df_modified["playlist_followers_track"]

df2 = df_modified.sort_values("playlist_followers_track", ascending = False)
df2["track_name"].head(6790).nunique()
``` 
**Bivariante Analysis:**
- We visualized the playlist number of followers vs playlist the number of albums and we found that the playlists with 50 albums has the most followers:
```
sns.lineplot(data=df_modified, x = "num_albums", y = "num_followers")
``` 
**Mutivariante Analysis:**
- We visualized 3D Scatter Plot for # Artists vs. # Albums vs. # Tracks. and found that there is a hight variablility between # Artists and # Albums, that's the reason why we gave them a high weight in compare with the # tracks: 
```
sns.lineplot(data=df_modified, x = "num_albums", y = "num_followers")
``` 

## Model Building

#### Clustering
- We divided the tracks into 41 cluster (as there are 41 existed track genres) and recommend to the user according to `cluster` and `popularity`.
- Very fast algorithm, and it has a dataset of 90k songs
##### Choosing the features 
- **Correlation** is used on reducing the tracks dataset features to the most important one.
- **PCA** is used on reducing the most important features into other pca components that maintains the highest variability of the data.
#### Content-Based Filtering
- We used track features and created the track `metadata` feature which includes the the `artist + album + track` name and got the mean of the similarity (using cosaine similarity) due to track features and track metadata feature. and recommend to the user according to `similarity` and `popularity`.
- We only sliced the most 1000 popular tracks to make the loading speed reasonable
  - As it takes 20 minutes to iterate over all the 90k tracks
  - Very accurate algorithm, and it has a dataset of 5k songs

## Recommendations
- We can improve our content-based recommendation algorithm
- We can recommend the songs relative to the song name, artist instead of the song name only to get more accurate result.


## Deployment
you can access our app by following this link [Spotify-Recommendation-System-Website](https://recommendify01.herokuapp.com/)
### Heroku
We deploy our flask app to [Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others. 
We prepared the needed files to deploy our app successfully:
- Procfile: contains run statements for app file and setup.sh.
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file (app.py)  successfully 
- model.py: contains the python code of the recommendation system algorithm.
### Flask 
We also create our app by using flask , then deployed it to Heroku . The files of this part are located into (Spotify-Recommendation-System-Website) folder. You can access the app by following this link : [Spotify-Recommendation-System-Website](https://recommendify01.herokuapp.com/)
