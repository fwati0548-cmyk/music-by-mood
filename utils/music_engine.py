"""
Music Recommendation Engine
Location: utils/music_engine.py
Target Data: data/music/dataset.csv
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
            # current_dir adalah folder 'utils'
            current_dir = os.path.dirname(__file__)
            
            # Keluar dari utils (..), masuk ke data/music
            data_dir = os.path.normpath(os.path.join(current_dir, "..", "data", "music"))
            dataset_path = os.path.join(data_dir, "dataset.csv")

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"File tidak ditemukan di: {dataset_path}")

            raw_df = pd.read_csv(dataset_path)

            # 1. DEDUPLIKASI GLOBAL (Penting: Nama + Artis harus unik)
            # Menjamin lagu seperti 'La Bachata' hanya muncul 1x meski punya banyak genre
            _self.df = (
                raw_df.sort_values('popularity', ascending=False)
                      .drop_duplicates(subset=['track_name', 'artists'], keep='first')
                      .reset_index(drop=True)
                      .copy()
            )

            # 2. LOAD MODEL & ENCODER (Lokasi sesuai struktur Anda)
            try:
                model_path = os.path.join(data_dir, "music_mood_model.pkl")
                encoder_path = os.path.join(data_dir, "label_encoder.pkl")
                _self.model = joblib.load(model_path)
                _self.label_encoder = joblib.load(encoder_path)
            except:
                _self.model = None
                _self.label_encoder = None

            # 3. PROSES MOOD
            _self._add_mood_column()
            
            # 4. LIST GENRE DARI DATA YANG SUDAH BERSIH
            _self.genres = sorted(_self.df['track_genre'].unique().tolist())
            
        except Exception as e:
            raise RuntimeError(f"Gagal memuat data. Error: {e}")

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
                self.df['mood'] = self.label_encoder.inverse_transform(self.model.predict(self.df[features]))
                return
            except: pass
        self.df['mood'] = self.df.apply(self._classify_mood_rule_based, axis=1)

    # ===============================================================
    # REKOMENDASI: ACAK TOTAL (TIDAK BOLEH PAKAI SORT DI AKHIR)
    # ===============================================================
    # === TAMBAHKAN KODE INI DI DALAM CLASS MusicRecommendationEngine ===

    def get_mood_distribution(self):
        """Menghitung jumlah lagu per mood untuk statistik dan Pie Chart"""
        if self.df is not None:
            return self.df['mood'].value_counts().to_dict()
        return {mood: 0 for mood in self.moods}

    def get_genre_distribution(self, mood=None):
        """Menghitung distribusi genre untuk Bar Chart"""
        df_to_use = self.df
        if mood and mood != "All Moods":
            df_to_use = self.df[self.df['mood'] == mood]
        
        if df_to_use is not None:
            # Mengambil 10 genre teratas agar grafik tidak berantakan
            return df_to_use['track_genre'].value_counts().to_dict()
        return {}

    def get_mood_stats(self):
        """Menghitung rata-rata fitur audio per mood untuk Radar Chart"""
        if self.df is not None:
            features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
            # Mengelompokkan berdasarkan mood dan mengambil rata-rata fitur audionya
            stats = self.df.groupby('mood')[features].mean()
            return stats
        return pd.DataFrame()
        
    def get_recommendations_by_mood(self, mood, n=10):
        filtered = self.df[self.df['mood'] == mood].copy()
        if filtered.empty: return pd.DataFrame()
        
        # Ambil n sampel secara acak dari seluruh dataset yang cocok
        # random_state=None memastikan hasil berbeda setiap kali diklik
        return filtered.sample(n=min(n, len(filtered)), random_state=None)[self._output_columns()]

    def get_recommendations_by_genre(self, genre, n=10):
        filtered = self.df[self.df['track_genre'] == genre].copy()
        if filtered.empty: return pd.DataFrame()
        return filtered.sample(n=min(n, len(filtered)), random_state=None)[self._output_columns()]

    def get_recommendations_by_mood_and_genre(self, mood, genre, n=10):
        filtered = self.df[(self.df['mood'] == mood) & (self.df['track_genre'] == genre)].copy()
        if filtered.empty: return pd.DataFrame()
        return filtered.sample(n=min(n, len(filtered)), random_state=None)[self._output_columns()]

    # ===============================================================
    # HELPERS
    # ===============================================================

    @staticmethod
    def create_spotify_embed(track_id, width=300, height=380):
        return f"""
        <iframe src="https://open.spotify.com/embed/track/{track_id}"
                width="{width}" height="{height}"
                frameborder="0" allowtransparency="true" allow="encrypted-media">
        </iframe>
        """

    def _output_columns(self):
        return ['track_name', 'artists', 'album_name', 'track_id', 'popularity', 'track_genre', 'mood']

    def get_available_genres(self): return self.genres
    def get_available_moods(self): return self.moods
