import cv2
import sqlite3
import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_ADMIN_ID = os.getenv('TELEGRAM_ADMIN_ID')
TIME_LIMIT_MINUTES = int(os.getenv('TIME_LIMIT_MINUTES', '5'))
TIME_LIMIT_SECONDS = TIME_LIMIT_MINUTES * 60

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT UNIQUE,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    total_time INTEGER,
    alert_sent INTEGER DEFAULT 0
)
''')
conn.commit()

face_trackers = {}
next_face_id = 1

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def match_face_to_tracker(face_box, trackers, threshold=0.3):
    best_match = None
    best_iou = threshold
    
    for face_id, tracker_data in trackers.items():
        last_box = tracker_data['last_box']
        iou = calculate_iou(face_box, last_box)
        if iou > best_iou:
            best_iou = iou
            best_match = face_id
    
    return best_match

def send_telegram_photo(photo_path, face_id):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    with open(photo_path, 'rb') as photo:
        files = {'photo': photo}
        data = {
            'chat_id': TELEGRAM_ADMIN_ID,
            'caption': f'Лицо {face_id} превысило лимит времени ({TIME_LIMIT_MINUTES} минут)'
        }
        try:
            response = requests.post(url, files=files, data=data, timeout=10)
            return response.status_code == 200
        except:
            return False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

frame_count = 0
detection_interval = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_time = time.time()
    
    if frame_count % detection_interval == 0:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        active_face_ids = set()
        
        for (x, y, w, h) in faces:
            face_box = (x, y, w, h)
            matched_id = match_face_to_tracker(face_box, face_trackers)
            
            if matched_id is None:
                face_id = f"face_{next_face_id}"
                next_face_id += 1
                face_trackers[face_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_box': face_box,
                    'alert_sent': False
                }
                
                cursor.execute('''
                    INSERT OR IGNORE INTO faces (face_id, first_seen, last_seen, total_time)
                    VALUES (?, ?, ?, ?)
                ''', (face_id, datetime.fromtimestamp(current_time), 
                      datetime.fromtimestamp(current_time), 0))
                conn.commit()
            else:
                face_trackers[matched_id]['last_seen'] = current_time
                face_trackers[matched_id]['last_box'] = face_box
                active_face_ids.add(matched_id)
        
        to_remove = []
        for face_id in face_trackers.keys():
            if face_id not in active_face_ids:
                time_diff = current_time - face_trackers[face_id]['last_seen']
                if time_diff > 2.0:
                    to_remove.append(face_id)
        
        for face_id in to_remove:
            del face_trackers[face_id]
    
    for face_id, tracker_data in face_trackers.items():
        x, y, w, h = tracker_data['last_box']
        elapsed_time = current_time - tracker_data['first_seen']
        
        cursor.execute('''
            UPDATE faces 
            SET last_seen = ?, total_time = ?
            WHERE face_id = ?
        ''', (datetime.fromtimestamp(current_time), int(elapsed_time), face_id))
        conn.commit()
        
        if elapsed_time > TIME_LIMIT_SECONDS:
            color = (0, 0, 255)
            thickness = 3
            
            if not tracker_data['alert_sent']:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    temp_path = f"temp_{face_id}.jpg"
                    cv2.imwrite(temp_path, face_roi)
                    
                    if send_telegram_photo(temp_path, face_id):
                        tracker_data['alert_sent'] = True
                        cursor.execute('''
                            UPDATE faces SET alert_sent = 1 WHERE face_id = ?
                        ''', (face_id,))
                        conn.commit()
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            color = (0, 255, 0)
            thickness = 2
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, f"{face_id} ({int(elapsed_time)}s)", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    person_count = len(face_trackers)
    cv2.putText(frame, f"Людей в кадре: {person_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
