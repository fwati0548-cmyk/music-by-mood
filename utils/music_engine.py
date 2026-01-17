import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st

class MusicRecommendationEngine:
    def __init__(self):
        self.df = None
        self.model = None
        self.label_encoder = None
        self.genres = []
        self.moods = ['Happy', 'Sad', 'Calm', 'Tense']
        self._load_data()

    @st.cache_resource
    def _load_data(_self):
        try:
            current_dir = os.path.dirname(__file__)
            data_dir = os.path.join(current_dir, "..", "data", "music")
            dataset_path = os.path.join(data_dir, "dataset.csv")
            
            raw_df = pd.read_csv(dataset_path)

            # DEDUPLIKASI GLOBAL (Agar tidak ada lagu dobel walau beda genre)
            _self.df = (
                raw_df.sort_values('popularity', ascending=False)
                      .drop_duplicates(subset=['track_name', 'artists'], keep='first')
                      .reset_index(drop=True)
                      .copy()
            )

            try:
                model_path = os.path.join(data_dir, "music_mood_model.pkl")
                encoder_path = os.path.join(data_dir, "label_encoder.pkl")
                _self.model = joblib.load(model_path)
                _self.label_encoder = joblib.load(encoder_path)
            except Exception:
                _self.model = None
                _self.label_encoder = None

            _self._add_mood_column()
            _self.genres = sorted(_self.df['track_genre'].unique().tolist())
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def _classify_mood_rule_based(self, row):
        v, e = row['valence'], row['energy']
        if v >= 0.5 and e >= 0.5: return 'Happy'
        elif v < 0.5 and e < 0.5: return 'Sad'
        elif v >= 0.5 and e < 0.5: return 'Calm'
        else: return 'Tense'

    def _add_mood_column(self):
        if self.model and self.label_encoder:
            try:
                features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'loudness', 'speechiness']
                self.df['mood'] = self.label_encoder.inverse_transform(self.model.predict(self.df[features]))
                return
            except: pass
        self.df['mood'] = self.df.apply(self._classify_mood_rule_based, axis=1)

    # --- RECOMMENDATION METHODS ---
    def get_recommendations_by_mood(self, mood, n=10):
        filtered = self.df[self.df['mood'] == mood].copy()
        if filtered.empty: return pd.DataFrame()
        pool = filtered.head(min(len(filtered), 50))
        return pool.sample(n=min(n, len(pool))).sort_values(by='popularity', ascending=False)[self._output_columns()]

    def get_recommendations_by_genre(self, genre, n=10):
        filtered = self.df[self.df['track_genre'] == genre].copy()
        if filtered.empty: return pd.DataFrame()
        pool = filtered.head(min(len(filtered), 50))
        return pool.sample(n=min(n, len(pool))).sort_values(by='popularity', ascending=False)[self._output_columns()]

    def get_recommendations_by_mood_and_genre(self, mood, genre, n=10):
        filtered = self.df[(self.df['mood'] == mood) & (self.df['track_genre'] == genre)].copy()
        if filtered.empty: return pd.DataFrame()
        return filtered.sample(n=min(n, len(filtered))).sort_values(by='popularity', ascending=False)[self._output_columns()]

    # --- HELPERS (PASTIKAN ADA DAN @STATICMETHOD) ---
    @staticmethod
    def create_spotify_embed(track_id, width=300, height=380):
        return f"""
        <iframe src="https://open.spotify.com/embed/track/{track_id}"
                width="{width}" height="{height}"
                frameborder="0" allow="encrypted-media">
        </iframe>
        """

    @staticmethod
    def create_spotify_search_link(track_name, artist):
        query = f"{track_name} {artist}".replace(" ", "+")
        return f"https://open.spotify.com/search/{query}"

    def _output_columns(self):
        return ['track_name', 'artists', 'album_name', 'track_id', 'popularity', 'track_genre', 'mood']

    def get_available_genres(self): return self.genres
    def get_available_moods(self): return self.moods
