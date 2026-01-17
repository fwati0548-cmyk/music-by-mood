"""
Music Recommendation Engine - VERSION 2.0
Fixed: No duplicate songs with different genres.
Fixed: AttributeError on create_spotify_embed.
"""

import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st

class MusicRecommendationEngine:
    """
    Modular music recommendation engine.
    Deduplicated by track_name and artists to ensure unique recommendations.
    """

    def __init__(self):
        self.df = None
        self.model = None
        self.label_encoder = None
        self.genres = []
        self.moods = ['Happy', 'Sad', 'Calm', 'Tense']
        self._load_data()

    # ===============================
    # LOAD DATA & DEDUPLICATION
    # ===============================
    @st.cache_resource
    def _load_data(_self):
        try:
            current_dir = os.path.dirname(__file__)
            # Menyesuaikan path data Anda
            data_dir = os.path.join(current_dir, "..", "data", "music")
            dataset_path = os.path.join(data_dir, "dataset.csv")
            
            # 1. Load raw data
            raw_df = pd.read_csv(dataset_path)

            # 2. GLOBAL DEDUPLICATION (PERBAIKAN UTAMA)
            # Kita urutkan berdasarkan popularity tertinggi, lalu hapus duplikat
            # berdasarkan track_name dan artists. Ini akan menyaring lagu yang sama
            # yang muncul di genre berbeda (misal: latin vs latino).
            _self.df = (
                raw_df.sort_values('popularity', ascending=False)
                      .drop_duplicates(subset=['track_name', 'artists'], keep='first')
                      .reset_index(drop=True)
                      .copy()
            )

            # 3. Load trained model (optional)
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
            print(f"🎵 Dataset Ready: {len(_self.df)} unique songs.")

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

        filtered = self.df[self.df['mood'] == mood].copy()
        if filtered.empty:
            return pd.DataFrame()

        # Keamanan tambahan: pastikan tetap unik
        filtered = filtered.drop_duplicates(subset=['track_name', 'artists'], keep='first')

        # Ambil dari pool lagu populer agar kualitas saran terjaga
        pool_size = min(len(filtered), max(n * 3, 50))
        top_pool = filtered.head(pool_size)

        recommendations = (
            top_pool.sample(n=min(n, len(top_pool)))
            .sort_values(by='popularity', ascending=False)
        )

        return recommendations[self._output_columns()]

    def get_recommendations_by_genre(self, genre, n=10):
        filtered = self.df[self.df['track_genre'] == genre].copy()
        if filtered.empty:
            return pd.DataFrame()

        filtered = filtered.drop_duplicates(subset=['track_name', 'artists'], keep='first')

        pool_size = min(len(filtered), max(n * 3, 50))
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
        ].copy()

        if filtered.empty:
            return pd.DataFrame()

        filtered = filtered.drop_duplicates(subset=['track_name', 'artists'], keep='first')

        recommendations = (
            filtered.sample(n=min(n, len(filtered)))
            .sort_values(by='popularity', ascending=False)
        )

        return recommendations[self._output_columns()]

    # ===============================
    # SPOTIFY HELPERS (FIXED ERROR)
    # ===============================
    def create_spotify_embed(self, track_id, width=300, height=380):
        """
        Membuat iframe untuk Spotify Embed.
        Menggunakan instance method agar bisa dipanggil lewat engine.create_spotify_embed
        """
        return f"""
        <iframe src="https://open.spotify.com/embed/track/{track_id}"
                width="{width}" height="{height}"
                frameborder="0" allow="encrypted-media">
        </iframe>
        """

    def create_spotify_search_link(self, track_name, artist):
        query = f"{track_name} {artist}".replace(" ", "+")
        return f"https://open.spotify.com/search/{query}"

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
