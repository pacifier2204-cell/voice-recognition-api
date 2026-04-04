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

    if database is None:
        print("No embeddings found.")
    else:
        print("Embeddings loaded successfully")


load_model()


# ================================
# REGISTER VOICE (FINAL FIXED)
# ================================
@app.post("/register_voice")
async def register_voice(username: str, files: List[UploadFile] = File(...)):

    global database

    sample_count = 0
    trained = False

    for file in files:

        # ✅ KEEP ORIGINAL FORMAT (IMPORTANT FIX)
        temp_file = f"temp_{uuid.uuid4()}_{file.filename}"

        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        database, sample_count, trained = add_new_user(username, temp_file)

        os.remove(temp_file)

    # ✅ RELOAD DATABASE AFTER TRAINING
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

    if database is None:
        raise HTTPException(status_code=400, detail="Model not ready.")

    # ✅ KEEP ORIGINAL FORMAT
    temp_file = f"temp_{uuid.uuid4()}_{file.filename}"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    user, confidence, margin = recognize_speaker(temp_file, database)

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

    os.remove(temp_file)

    return {
        "recognized_user": user,
        "confidence": confidence,
        "status": status
    }