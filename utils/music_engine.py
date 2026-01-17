"""
Music Recommendation Engine
Handles music data loading, mood classification, and recommendations
(FIXED: Menghapus duplikat lagu yang memiliki genre berbeda-beda)
"""

import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st

class MusicRecommendationEngine:
    """
    Modular music recommendation engine
    Sistem ini memfilter lagu agar satu lagu hanya muncul satu kali meskipun memiliki banyak genre.
    """

    def __init__(self):
        self.df = None
        self.model = None
        self.label_encoder = None
        self.genres = []
        self.moods = ['Happy', 'Sad', 'Calm', 'Tense']
        self._load_data()

    # ===============================
    # LOAD DATA & FILTER DUPLIKAT
    # ===============================
    @st.cache_resource
    def _load_data(_self):
        try:
            current_dir = os.path.dirname(__file__)
            data_dir = os.path.join(current_dir, "..", "data", "music")
            dataset_path = os.path.join(data_dir, "dataset.csv")
            
            # 1. Load dataset asli
            raw_df = pd.read_csv(dataset_path)

            # 2. PROSES FILTER UTAMA:
            # Urutkan berdasarkan popularitas tertinggi, lalu hapus duplikat
            # berdasarkan Nama Lagu dan Artis. Ini memastikan jika satu lagu punya 
            # 5 genre berbeda, hanya 1 genre (dari baris paling populer) yang diambil.
            _self.df = (
                raw_df.sort_values('popularity', ascending=False)
                      .drop_duplicates(subset=['track_name', 'artists'], keep='first')
                      .copy()
            )

            # 3. Load trained model (jika ada)
            try:
                model_path = os.path.join(data_dir, "music_mood_model.pkl")
                encoder_path = os.path.join(data_dir, "label_encoder.pkl")

                _self.model = joblib.load(model_path)
                _self.label_encoder = joblib.load(encoder_path)
                print("✅ Trained models loaded")
            except Exception:
                _self.model = None
                _self.label_encoder = None
                print("⚠️ Using rule-based mood classification")

            # 4. Tambahkan kolom mood pada data yang sudah bersih
            _self._add_mood_column()

            # 5. Ambil daftar genre unik dari data yang sudah difilter
            _self.genres = sorted(_self.df['track_genre'].unique().tolist())
            print(f"🎵 Dataset Ready: {len(_self.df)} lagu unik dimuat.")

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    # ===============================
    # MOOD CLASSIFICATION
    # ===============================
    def _classify_mood_rule_based(self, row):
        valence = row['valence']
        energy = row['energy']

        if valence >= 0.5 and energy >= 0.5:
            return 'Happy'
        elif valence < 0.5 and energy < 0.5:
            return 'Sad'
        elif valence >= 0.5 and energy < 0.5:
            return 'Calm'
        else:
            return 'Tense'

    def _add_mood_column(self):
        if self.model is not None and self.label_encoder is not None:
            try:
                features = ['danceability', 'energy', 'valence', 'tempo', 
                            'acousticness', 'instrumentalness', 'loudness', 'speechiness']
                X = self.df[features]
                mood_encoded = self.model.predict(X)
                self.df['mood'] = self.label_encoder.inverse_transform(mood_encoded)
                return
            except Exception:
                pass

        self.df['mood'] = self.df.apply(self._classify_mood_rule_based, axis=1)

    # ===============================
    # RECOMMENDATION METHODS
    # ===============================
    def get_recommendations_by_mood(self, mood, n=10):
        if mood not in self.moods:
            raise ValueError(f"Mood must be one of {self.moods}")

        # Filter berdasarkan mood (Data sudah pasti unik karena proses di awal)
        filtered = self.df[self.df['mood'] == mood]
        
        if filtered.empty:
            return pd.DataFrame()

        # Ambil sampel secara acak dari pool lagu terpopuler
        pool_size = min(len(filtered), max(n * 2, n))
        top_pool = filtered.head(pool_size)

        recommendations = (
            top_pool.sample(n=min(n, len(top_pool)))
            .sort_values(by='popularity', ascending=False)
        )

        return recommendations[self._output_columns()]

    def get_recommendations_by_genre(self, genre, n=10):
        filtered = self.df[self.df['track_genre'] == genre]
        
        if filtered.empty:
            return pd.DataFrame()

        pool_size = min(len(filtered), max(n * 2, n))
        top_pool = filtered.head(pool_size)

        recommendations = (
            top_pool.sample(n=min(n, len(top_pool)))
            .sort_values(by='popularity', ascending=False)
        )

        return recommendations[self._output_columns()]

    def get_recommendations_by_mood_and_genre(self, mood, genre, n=10):
        filtered = self.df[
            (self.df['mood'] == mood) & 
            (self.df['track_genre'] == genre)
        ]

        if filtered.empty:
            return pd.DataFrame()

        recommendations = (
            filtered.sample(n=min(n, len(filtered)))
            .sort_values(by='popularity', ascending=False)
        )

        return recommendations[self._output_columns()]

    # ===============================
    # UTILITIES
    # ===============================
    def _output_columns(self):
        return [
            'track_name', 'artists', 'album_name', 
            'track_id', 'popularity', 'track_genre', 'mood'
        ]

    def get_available_genres(self):
        return self.genres

    def get_available_moods(self):
        return self.moods
