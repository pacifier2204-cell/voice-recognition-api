import os
import numpy as np
import librosa
from resemblyzer import VoiceEncoder

import firebase_admin
from firebase_admin import credentials, firestore

# ================================
# FIREBASE INIT
# ================================
cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
encoder = VoiceEncoder()

# Training rules
MIN_TRAINING_SAMPLES = 15
MAX_STORED_SAMPLES = 25


# ================================
# LOAD DATABASE
# ================================
def load_database():
    users = db.collection("voice_embeddings").stream()
    database = {}

    for user in users:
        data = user.to_dict()

        if not data.get("trained", False):
            continue

        embeddings = data["embeddings"]
        database[user.id] = [np.array(e) for e in embeddings]

    if len(database) == 0:
        return None

    return database


# ================================
# ADD NEW USER / TRAINING
# ================================
def add_new_user(username, file_path):

    # ✅ Supports any audio format
    wav, sr = librosa.load(file_path, sr=16000)
    embed = encoder.embed_utterance(wav)

    user_ref = db.collection("voice_embeddings").document(username)
    doc = user_ref.get()

    if doc.exists:
        data = doc.to_dict()
        embeddings = data.get("embeddings", [])
        sample_count = data.get("sample_count", 0)
    else:
        embeddings = []
        sample_count = 0

    embeddings.append(embed.tolist())
    sample_count += 1

    if len(embeddings) > MAX_STORED_SAMPLES:
        embeddings = embeddings[-MAX_STORED_SAMPLES:]

    trained = sample_count >= MIN_TRAINING_SAMPLES

    user_ref.set({
        "embeddings": embeddings,
        "sample_count": sample_count,
        "trained": trained
    })

    return load_database(), sample_count, trained


# ================================
# SPEAKER RECOGNITION
# ================================
def recognize_speaker(file_path, database):

    # ✅ Supports any audio format
    wav, sr = librosa.load(file_path, sr=16000)

    if len(wav) < 16000:
        return "Invalid", 0.0, 0.0

    embed = encoder.embed_utterance(wav)

    user_scores = {}

    for user, embeddings in database.items():

        scores = []

        for db_embed in embeddings:
            score = np.dot(embed, db_embed) / (
                np.linalg.norm(embed) * np.linalg.norm(db_embed)
            )
            scores.append(score)

        top_k = sorted(scores, reverse=True)[:3]
        user_scores[user] = np.mean(top_k)

    # ✅ Prevent crash
    if not user_scores:
        return "Unknown", 0.0, 0.0

    sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

    best_user, best_score = sorted_users[0]
    second_score = sorted_users[1][1] if len(sorted_users) > 1 else 0

    margin = best_score - second_score

    return best_user, float(best_score), float(margin)