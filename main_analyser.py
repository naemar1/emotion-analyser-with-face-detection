"""import cv2
import face_recognition
import numpy as np
import pyttsx3
import threading
import speech_recognition as sr
import sqlite3
import os
from tensorflow.keras.models import load_model

# --- 1. DATABASE & CACHE ---
conn = sqlite3.connect('user_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users (name TEXT, encoding BLOB)')
conn.commit()

# Load memory cache once to stop lag
c.execute("SELECT name, encoding FROM users")
db_rows = c.fetchall()
known_names = [r[0] for r in db_rows]
known_encs = [np.frombuffer(r[1], dtype=np.float64) for r in db_rows]

# --- 2. THREADED VOICE ---
def speak(text):
    def _speak_thread():
        try:
            engine = pyttsx3.init(driverName='sapi5')
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except: pass
    threading.Thread(target=_speak_thread, daemon=True).start()

# --- 3. SPEECH RECOGNITION ---
def listen_for_name(prompt):
    speak(prompt)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.8)
        try:
            print("Listening...")
            audio = r.listen(source, timeout=6, phrase_time_limit=4)
            text = r.recognize_google(audio).lower()
            return text.replace(" ", "_")
        except:
            return None

# --- 4. LOAD MODELS ---
try:
    face_model = load_model('facial_model.h5')
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
except:
    emotions = ['Scanning...']

# --- 5. MAIN LOOP ---
cap = cv2.VideoCapture(0)
frame_count = 0
skip_frames = 6
last_face_mood = "Neutral"

# FIX: Initialize these variables so the loop doesn't crash on frame 1
face_locs = []
face_encs = []
target_name = "Stranger"
target_enc = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    h, w, _ = frame.shape

    # UI COMMAND BAR (Center Bottom, One Line)
    cmd_txt = "[S] ANALYZE/REG    |    [E] EDIT NAME    |    [Q] SHUT DOWN"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(cmd_txt, font, 0.5, 1)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - tw//2 - 20, h - 60), (w//2 + tw//2 + 20, h - 20), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, cmd_txt, (w//2 - tw//2, h - 35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Recognition Logic
    if frame_count % skip_frames == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    # Reset target identity for this frame
    target_name = "Stranger"
    target_enc = None

    for i, (top, right, bottom, left) in enumerate(face_locs):
        t, r, b, l = top*4, right*4, bottom*4, left*4
        name = "Stranger"
        
        if len(known_encs) > 0 and i < len(face_encs):
            # Adjusted tolerance to 0.54 (Balanced)
            matches = face_recognition.compare_faces(known_encs, face_encs[i], tolerance=0.54)
            face_distances = face_recognition.face_distance(known_encs, face_encs[i])
            
            if len(face_distances) > 0:
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    name = known_names[best_match_idx]

        # Emotion Recognition
        if frame_count % skip_frames == 0:
            roi = cv2.cvtColor(frame[t:b, l:r], cv2.COLOR_BGR2GRAY)
            if roi.size != 0:
                roi = cv2.resize(roi, (48, 48)) / 255.0
                pred = face_model.predict(roi.reshape(1, 48, 48, 1), verbose=0)
                last_face_mood = emotions[np.argmax(pred)]

        # Visuals
        color = (0, 255, 0) if name != "Stranger" else (0, 0, 255)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"{name.replace('_',' ')}: {last_face_mood}", (l, t-10), font, 0.7, color, 2)
        
        # Set first face as target
        if i == 0:
            target_name = name
            target_enc = face_encs[i]

    cv2.imshow('AI Assistant', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if target_name == "Stranger" and target_enc is not None:
            res = listen_for_name("What is your name?Either say it normally or spell it out for me.")
            if res:
                c.execute("INSERT INTO users VALUES (?, ?)", (res, target_enc.tobytes()))
                conn.commit()
                known_names.append(res)
                known_encs.append(target_enc)
                speak(f"Registered {res}")
        elif target_name != "Stranger":
            speak(f"Hello {target_name}. You seem {last_face_mood}.")

    elif key == ord('e') and target_name != "Stranger":
        upd = listen_for_name(f"What should I call you instead of {target_name.replace('_',' ')}? Please the say and spelling if needed.")
        if upd:
            c.execute("UPDATE users SET name = ? WHERE name = ?", (upd, target_name))
            conn.commit()
            idx = known_names.index(target_name)
            known_names[idx] = upd
            speak(f"Your name has been updated. Hello {upd.replace('_',' ')}.")

    elif key == ord('q'):
        speak("Thank you for using this system.. Bye! Have a nice day... ")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()"""


import cv2
import face_recognition
import numpy as np
import pyttsx3
import threading
import sqlite3
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from tensorflow.keras.models import load_model

# --- 1. DATABASE & CACHE ---
conn = sqlite3.connect('user_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users (name TEXT, encoding BLOB)')
conn.commit()

def load_cache():
    c.execute("SELECT name, encoding FROM users")
    db_rows = c.fetchall()
    names = [r[0] for r in db_rows]
    encs = [np.frombuffer(r[1], dtype=np.float64) for r in db_rows]
    return names, encs

known_names, known_encs = load_cache()

# --- 2. THREADED VOICE ---
def speak(text):
    def _speak_thread():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except: pass
    threading.Thread(target=_speak_thread, daemon=True).start()

# --- 3. GUI UTILS ---
def get_name_popup(prompt_text):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    name = simpledialog.askstring("Input", prompt_text, parent=root)
    root.destroy()
    return name.strip().replace(" ", "_") if name else None

def confirm_delete(name):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    res = messagebox.askyesno("Delete User", f"Are you sure you want to delete {name.replace('_',' ')}?")
    root.destroy()
    return res

# --- 4. LOAD MODELS ---
try:
    face_model = load_model('facial_model.h5')
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
except:
    emotions = ['Scanning...']

# --- 5. MAIN LOOP ---
cap = cv2.VideoCapture(0)
frame_count = 0
skip_frames = 6
last_face_mood = "Neutral"
face_locs, face_encs = [], []
target_name, target_enc = "Stranger", None

speak("System is active. Press S to analyze or register, E to edit name, D to delete user, Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    h, w, _ = frame.shape

    # UI COMMAND BAR
    cmd_txt = "[S] REG/ANALYZE  |  [E] EDIT  |  [D] DELETE  |  [Q] QUIT"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(cmd_txt, font, 0.5, 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - tw//2 - 20, h - 60), (w//2 + tw//2 + 20, h - 20), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, cmd_txt, (w//2 - tw//2, h - 35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if frame_count % skip_frames == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    current_target_name = "Stranger"
    current_target_enc = None

    for i, (top, right, bottom, left) in enumerate(face_locs):
        t, r, b, l = top*4, right*4, bottom*4, left*4
        name = "Stranger"
        
        if len(known_encs) > 0 and i < len(face_encs):
            face_distances = face_recognition.face_distance(known_encs, face_encs[i])
            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if face_distances[best_idx] < 0.54:
                    name = known_names[best_idx]

        if frame_count % skip_frames == 0:
            roi = cv2.cvtColor(frame[t:b, l:r], cv2.COLOR_BGR2GRAY)
            if roi.size != 0:
                roi = cv2.resize(roi, (48, 48)) / 255.0
                pred = face_model.predict(roi.reshape(1, 48, 48, 1), verbose=0)
                last_face_mood = emotions[np.argmax(pred)]

        color = (0, 255, 0) if name != "Stranger" else (0, 0, 255)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"{name.replace('_',' ')}: {last_face_mood}", (l, t-10), font, 0.7, color, 2)
        
        if i == 0:
            current_target_name = name
            current_target_enc = face_encs[i]

    cv2.imshow('AI Assistant', frame)
    
    key = cv2.waitKey(1) & 0xFF
    # REGISTRATION / GREETING
    if key == ord('s'):
        if current_target_name == "Stranger" and current_target_enc is not None:
            speak("I don't recognize you. Let's get you registered.")
            res = get_name_popup("Unknown face. Enter name to register:")
            if res:
                c.execute("INSERT INTO users VALUES (?, ?)", (res, current_target_enc.tobytes()))
                conn.commit()
                known_names.append(res); known_encs.append(current_target_enc)
                speak(f"Registered {res}")
        elif current_target_name != "Stranger":
            speak(f"Hello {current_target_name.replace('_',' ')}. You seem {last_face_mood}.")

    # EDIT NAME
    elif key == ord('e') and current_target_name != "Stranger":
        upd = get_name_popup(f"Changing name for {current_target_name.replace('_',' ')} to:")
        if upd:
            c.execute("UPDATE users SET name = ? WHERE name = ?", (upd, current_target_name))
            conn.commit()
            known_names[known_names.index(current_target_name)] = upd
            speak(f"Updated to {upd.replace('_',' ')}.")

    # DELETE USER
    elif key == ord('d') and current_target_name != "Stranger":
        if confirm_delete(current_target_name):
            c.execute("DELETE FROM users WHERE name = ?", (current_target_name,))
            conn.commit()
            idx = known_names.index(current_target_name)
            known_names.pop(idx); known_encs.pop(idx)
            speak(f"Deleted {current_target_name.replace('_',' ')} from system.")

    elif key == ord('q'):
        speak("Thank you for using this system. Goodbye!")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()