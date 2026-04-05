import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException

from model.embedding_model import load_database, recognize_speaker, add_new_user

app = FastAPI()

database = None


# ================================
# LOAD MODEL
# ================================
def load_model():
    global database
    database = load_database()
    if not database:
        print("No embeddings found.")
    else:
        print(f"Embeddings loaded for {len(database)} users.")


load_model()

import subprocess

@app.get("/")
def health_check():
    return {
        "status": "online",
        "trained_users": len(database) if database else 0
    }

@app.get("/check_setup")
def check_setup():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ffmpeg_ok = "ffmpeg version" in result.stdout
    except:
        ffmpeg_ok = False

    try:
        from resemblyzer import VoiceEncoder
        resemblyzer_ok = True
    except:
        resemblyzer_ok = False

    try:
        docs = list(db.collection("voice_embeddings").limit(1).stream())
        firebase_ok = True
    except:
        firebase_ok = False

    return {
        "ffmpeg": ffmpeg_ok,
        "resemblyzer": resemblyzer_ok,
        "firebase": firebase_ok,
        "database_loaded": database is not None,
        "trained_users": len(database) if database else 0
    }

# ================================
# REGISTER VOICE
# ================================
@app.post("/register_voice")
async def register_voice(username: str, files: List[UploadFile] = File(...)):

    global database

    sample_count = 0
    trained = False
    temp_files = []

    try:
        for file in files:
            temp_file = f"temp_{uuid.uuid4()}_{file.filename}"
            temp_files.append(temp_file)

            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Call add_new_user per file — core logic unchanged
            _, sample_count, trained = add_new_user(username, temp_file)

    finally:
        # ✅ Always clean up temp files, even if an exception occurred
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

    # ✅ Single reload after all files in batch are processed
    database = load_database()

    return {
        "status": "training_complete" if trained else "training_in_progress",
        "user": username,
        "samples_recorded": sample_count,
        "samples_required": 15
    }


# ================================
# VOICE RECOGNITION
# ================================
@app.post("/recognize_voice")
async def recognize_voice(file: UploadFile = File(...)):

    global database

    # ✅ Treat empty dict same as None — server is ready, just no trained users
    if not database:
        raise HTTPException(status_code=400, detail="No trained voice profiles found.")

    temp_file = f"temp_{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        user, confidence, margin = recognize_speaker(temp_file, database)

    finally:
        # ✅ Always clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

    base_threshold = 0.88

    if margin < 0.03:
        base_threshold += 0.03
    elif margin > 0.10:
        base_threshold -= 0.02

    if confidence < 0.70:
        user = "Unknown"
        status = "not_recognized"

    elif confidence > base_threshold and margin > 0.08:
        status = "recognized"

    elif confidence > 0.70 and margin > 0.04:
        status = "low_confidence"

    else:
        user = "Unknown"
        status = "not_recognized"

    return {
        "recognized_user": user,
        "confidence": confidence,
        "status": status
    }