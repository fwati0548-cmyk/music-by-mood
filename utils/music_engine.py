"""
Music Recommendation Engine - FINAL FIX
Deduplication based on track_name and artists to stop repeating songs with different genres.
"""

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
            
            # Load data asli
            df_raw = pd.read_csv(dataset_path)

            # --- DEDUPLIKASI KETAT ---
            # Kita urutkan berdasarkan popularity dulu, lalu buang nama lagu + artis yang sama.
            # Ini akan menghapus "La Bachata" yang punya genre latin, latino, reggae, dll.
            _self.df = (
                df_raw.sort_values('popularity', ascending=False)
                      .drop_duplicates(subset=['track_name', 'artists'], keep='first')
                      .reset_index(drop=True)
            )

            # Load Model
            try:
                model_path = os.path.join(data_dir, "music_mood_model.pkl")
                encoder_path = os.path.join(data_dir, "label_encoder.pkl")
                _self.model = joblib.load(model_path)
                _self.label_encoder = joblib.load(encoder_path)
            except:
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
                features = ['danceability', 'energy', 'valence', 'tempo', 
                            'acousticness', 'instrumentalness', 'loudness', 'speechiness']
                X = self.df[features]
                preds = self.model.predict(X)
                self.df['mood'] = self.label_encoder.inverse_transform(preds)
                return
            except: pass
        self.df['mood'] = self.df.apply(self._classify_mood_rule_based, axis=1)

    def get_recommendations_by_mood(self, mood, n=10):
        if mood not in self.moods:
            raise ValueError(f"Mood must be {self.moods}")

        # Filter data yang sudah di-dedup di awal
        filtered = self.df[self.df['mood'] == mood].copy()
        
        if filtered.empty:
            return pd.DataFrame()

        # KEAMANAN TAMBAHAN: Lakukan dedup ulang sebelum sampling
        filtered = filtered.drop_duplicates(subset=['track_name', 'artists'], keep='first')

        # Ambil pool lagu terpopuler agar tidak random lagu tidak jelas
        pool_size = min(len(filtered), max(n * 3, 50))
        top_pool = filtered.head(pool_size)

        # Ambil n lagu secara acak dari pool terpopuler
        recommendations = (
            top_pool.sample(n=min(n, len(top_pool)))
            .sort_values(by='popularity', ascending=False)
        )

        return recommendations[self._output_columns()]

    def get_recommendations_by_genre(self, genre, n=10):
        filtered = self.df[self.df['track_genre'] == genre].copy()
        if filtered.empty: return pd.DataFrame()
        
        filtered = filtered.drop_duplicates(subset=['track_name', 'artists'], keep='first')
        
        pool_size = min(len(filtered), max(n * 3, 50))
        top_pool = filtered.head(pool_size)
        
        return top_pool.sample(n=min(n, len(top_pool))).sort_values(by='popularity', ascending=False)[self._output_columns()]

    def get_recommendations_by_mood_and_genre(self, mood, genre, n=10):
        filtered = self.df[(self.df['mood'] == mood) & (self.df['track_genre'] == genre)].copy()
        if filtered.empty: return pd.DataFrame()
        
        filtered = filtered.drop_duplicates(subset=['track_name', 'artists'], keep='first')
        
        count = min(n, len(filtered))
        return filtered.sample(n=count).sort_values(by='popularity', ascending=False)[self._output_columns()]

    def _output_columns(self):
        return ['track_name', 'artists', 'album_name', 'track_id', 'popularity', 'track_genre', 'mood']

    def get_available_genres(self): return self.genres
    def get_available_moods(self): return self.moods
