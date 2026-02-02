import cv2
import sqlite3
import os
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

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
    alert_sent INTEGER DEFAULT 0,
    face_image BLOB,
    face_encoding BLOB
)
''')

try:
    cursor.execute('ALTER TABLE faces ADD COLUMN face_image BLOB')
except:
    pass

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

def get_face_features(face_image):
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (100, 100))
    
    hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    lbp = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
    lbp = cv2.normalize(lbp, lbp).flatten()
    
    features = np.concatenate([hist, lbp])
    return features

def compare_faces(features1, features2, threshold=0.7):
    if features1 is None or features2 is None:
        return False
    
    correlation = cv2.compareHist(features1[:256].reshape(256, 1).astype(np.float32), 
                                   features2[:256].reshape(256, 1).astype(np.float32), 
                                   cv2.HISTCMP_CORREL)
    
    if correlation > threshold:
        return True
    
    diff = np.abs(features1 - features2)
    similarity = 1.0 - np.mean(diff)
    
    return similarity > threshold

def match_face_in_db(face_features):
    cursor.execute('SELECT face_id, face_image FROM faces WHERE face_image IS NOT NULL')
    known_faces = cursor.fetchall()
    
    for face_id, stored_image_bytes in known_faces:
        if stored_image_bytes:
            nparr = np.frombuffer(stored_image_bytes, np.uint8)
            stored_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if stored_image is not None and stored_image.size > 0:
                stored_features = get_face_features(stored_image)
                if compare_faces(face_features, stored_features):
                    return face_id
    return None

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

def put_text_ru(img, text, position, font_size, color_bgr, thickness):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color_rgb)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def send_telegram_message(text, chat_id=None):
    if chat_id is None:
        chat_id = TELEGRAM_ADMIN_ID
    print(f"[TELEGRAM] Sending message to chat_id: {chat_id}")
    print(f"[TELEGRAM] Message text: {text[:50]}...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id': chat_id,
        'text': text
    }
    try:
        response = requests.post(url, data=data, timeout=10)
        print(f"[TELEGRAM] Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[TELEGRAM] Error response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"[TELEGRAM] Exception: {e}")
        return False

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

def get_statistics():
    print("[STATS] Getting statistics...")
    stats_conn = sqlite3.connect('faces.db')
    stats_cursor = stats_conn.cursor()
    
    try:
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day, 0, 0, 0)
        hour_start = datetime(now.year, now.month, now.day, now.hour, 0, 0)
        
        today_start_str = today_start.strftime('%Y-%m-%d %H:%M:%S')
        hour_start_str = hour_start.strftime('%Y-%m-%d %H:%M:%S')
        today_date_str = today_start.strftime('%Y-%m-%d')
        
        print(f"[STATS] Today start: {today_start_str}")
        print(f"[STATS] Hour start: {hour_start_str}")
        print(f"[STATS] Today date: {today_date_str}")
        
        stats_cursor.execute('SELECT COUNT(*) FROM faces')
        total_faces = stats_cursor.fetchone()[0]
        print(f"[STATS] Total faces in DB: {total_faces}")
        
        try:
            stats_cursor.execute('''
                SELECT COUNT(DISTINCT face_id) 
                FROM faces 
                WHERE strftime('%Y-%m-%d', first_seen) = ?
            ''', (today_date_str,))
            result = stats_cursor.fetchone()
            print(f"[STATS] Daily query result: {result}")
            daily_count = result[0] if result and result[0] is not None else 0
            print(f"[STATS] Daily count: {daily_count}")
            
            if daily_count == 0:
                stats_cursor.execute('''
                    SELECT COUNT(DISTINCT face_id), MIN(first_seen), MAX(first_seen)
                    FROM faces
                ''')
                debug_result = stats_cursor.fetchone()
                print(f"[STATS] Debug - All faces: count={debug_result[0]}, min={debug_result[1]}, max={debug_result[2]}")
                
                stats_cursor.execute('''
                    SELECT face_id, first_seen 
                    FROM faces 
                    LIMIT 5
                ''')
                sample_faces = stats_cursor.fetchall()
                print(f"[STATS] Sample faces: {sample_faces}")
        except Exception as e:
            print(f"[STATS] Error in daily query: {e}")
            import traceback
            traceback.print_exc()
            daily_count = 0
        
        try:
            stats_cursor.execute('''
                SELECT COUNT(DISTINCT face_id) 
                FROM faces 
                WHERE datetime(first_seen) >= datetime(?)
            ''', (hour_start_str,))
            result = stats_cursor.fetchone()
            print(f"[STATS] Hourly query result: {result}")
            hourly_count = result[0] if result and result[0] is not None else 0
            print(f"[STATS] Hourly count: {hourly_count}")
        except Exception as e:
            print(f"[STATS] Error in hourly query: {e}")
            import traceback
            traceback.print_exc()
            hourly_count = 0
        
        print(f"[STATS] Returning: daily={daily_count}, hourly={hourly_count}")
        return daily_count, hourly_count
    finally:
        stats_conn.close()

def generate_report():
    print("[REPORT] Generating report...")
    report_conn = sqlite3.connect('faces.db')
    report_cursor = report_conn.cursor()
    
    try:
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day, 0, 0, 0)
        today_date_str = today_start.strftime('%Y-%m-%d')
        
        report_cursor.execute('''
            SELECT DISTINCT face_id, first_seen 
            FROM faces 
            WHERE strftime('%Y-%m-%d', first_seen) = ?
            ORDER BY first_seen
        ''', (today_date_str,))
        
        results = report_cursor.fetchall()
        print(f"[REPORT] Found {len(results)} unique visits today")
        
        all_hours = []
        for hour in range(24):
            hour_dt = today_start.replace(hour=hour, minute=0, second=0, microsecond=0)
            all_hours.append(hour_dt)
        
        hourly_visits = {h: 0 for h in all_hours}
        
        for row in results:
            face_id, first_seen = row
            visit_time = None
            
            if isinstance(first_seen, str):
                try:
                    if 'T' in first_seen or '+' in first_seen:
                        visit_time = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                    else:
                        visit_time = datetime.strptime(first_seen, '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"[REPORT] Error parsing date {first_seen}: {e}")
                    try:
                        visit_time = datetime.strptime(first_seen.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    except:
                        continue
            elif isinstance(first_seen, datetime):
                visit_time = first_seen
            else:
                print(f"[REPORT] Unknown date type: {type(first_seen)}, value: {first_seen}")
                continue
            
            if visit_time:
                hour_key = visit_time.replace(minute=0, second=0, microsecond=0)
                if hour_key in hourly_visits:
                    hourly_visits[hour_key] += 1
                else:
                    print(f"[REPORT] Hour {hour_key} not in range")
        
        hours = sorted(hourly_visits.keys())
        counts = [hourly_visits[h] for h in hours]
        
        print(f"[REPORT] Total hours: {len(hours)}")
        print(f"[REPORT] Total visits: {sum(counts)}")
        print(f"[REPORT] Sample counts: {counts[:5]}...")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.bar(range(24), counts, color='#2E86AB', alpha=0.7, edgecolor='#1a5f7a', linewidth=1)
        ax.plot(range(24), counts, marker='o', linewidth=2, markersize=6, color='#A23B72', label='Тренд')
        
        ax.set_title('Посещаемость по часам (сегодня)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Час дня', fontsize=12)
        ax.set_ylabel('Количество уникальных посетителей', fontsize=12)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend()
        
        for i, count in enumerate(counts):
            if count > 0:
                ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        report_path = 'report.png'
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[REPORT] Report saved to {report_path}")
        return report_path
        
    except Exception as e:
        print(f"[REPORT] Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        report_conn.close()

def send_telegram_document(file_path, chat_id):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    
    with open(file_path, 'rb') as document:
        files = {'document': document}
        data = {'chat_id': chat_id}
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            print(f"[TELEGRAM] Document response status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"[TELEGRAM] Exception sending document: {e}")
            return False

def check_telegram_commands():
    print("[BOT] Starting Telegram commands checker...")
    print(f"[BOT] Admin ID from env: {TELEGRAM_ADMIN_ID}")
    print(f"[BOT] Bot token exists: {bool(TELEGRAM_BOT_TOKEN)}")
    last_update_id = 0
    
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {'offset': last_update_id + 1, 'timeout': 10}
            print(f"[BOT] Requesting updates with offset: {last_update_id + 1}")
            response = requests.get(url, params=params, timeout=15)
            
            print(f"[BOT] Response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"[BOT] Response ok: {data.get('ok')}")
                print(f"[BOT] Results count: {len(data.get('result', []))}")
                
                if data.get('ok') and data.get('result'):
                    for update in data['result']:
                        last_update_id = update['update_id']
                        print(f"[BOT] Processing update_id: {last_update_id}")
                        
                        if 'message' in update:
                            message = update['message']
                            chat_id = message.get('chat', {}).get('id')
                            print(f"[BOT] Received message from chat_id: {chat_id}")
                            print(f"[BOT] Admin ID check: {str(chat_id)} == {str(TELEGRAM_ADMIN_ID)}")
                            
                            if str(chat_id) == str(TELEGRAM_ADMIN_ID):
                                print("[BOT] Chat ID matches admin ID")
                                if 'text' in message:
                                    text = message['text'].strip()
                                    print(f"[BOT] Command received: {text}")
                                    
                                    if text == '/start':
                                        print("[BOT] Processing /start command")
                                        welcome_text = "👋 Добро пожаловать!\n\n"
                                        welcome_text += "Я бот для мониторинга системы распознавания лиц.\n\n"
                                        welcome_text += "Доступные команды:\n"
                                        welcome_text += "/stat - получить статистику посещений\n"
                                        welcome_text += "/report - график посещаемости за сегодня"
                                        send_telegram_message(welcome_text, chat_id)
                                    
                                    elif text == '/stat':
                                        print("[BOT] Processing /stat command")
                                        daily, hourly = get_statistics()
                                        stat_text = "📊 Статистика:\n\n"
                                        stat_text += f"👥 За сегодня: {daily} человек\n"
                                        stat_text += f"⏰ За последний час: {hourly} человек"
                                        print(f"[BOT] Sending statistics: daily={daily}, hourly={hourly}")
                                        send_telegram_message(stat_text, chat_id)
                                    
                                    elif text == '/report':
                                        print("[BOT] Processing /report command")
                                        send_telegram_message("📈 Генерирую отчет...", chat_id)
                                        report_path = generate_report()
                                        if report_path and os.path.exists(report_path):
                                            if send_telegram_document(report_path, chat_id):
                                                print(f"[BOT] Report sent successfully")
                                                if os.path.exists(report_path):
                                                    os.remove(report_path)
                                            else:
                                                send_telegram_message("❌ Ошибка при отправке отчета", chat_id)
                                        else:
                                            send_telegram_message("❌ Нет данных для отчета за сегодня", chat_id)
                                else:
                                    print("[BOT] Message has no text field")
                            else:
                                print(f"[BOT] Chat ID {chat_id} does not match admin ID {TELEGRAM_ADMIN_ID}")
                        else:
                            print("[BOT] Update has no message field")
                else:
                    print("[BOT] No results in response")
            else:
                print(f"[BOT] Bad response: {response.text}")
        except Exception as e:
            print(f"[BOT] Exception in check_telegram_commands: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(2)

telegram_thread = threading.Thread(target=check_telegram_commands, daemon=True)
telegram_thread.start()

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
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    face_features = get_face_features(face_roi)
                    db_face_id = match_face_in_db(face_features)
                    
                    if db_face_id:
                        face_id = db_face_id
                        cursor.execute('SELECT first_seen, alert_sent FROM faces WHERE face_id = ?', (face_id,))
                        result = cursor.fetchone()
                        if result and result[0]:
                            if isinstance(result[0], str):
                                first_seen_dt = datetime.fromisoformat(result[0].replace('Z', '+00:00'))
                            else:
                                first_seen_dt = result[0]
                            first_seen_time = first_seen_dt.timestamp()
                            alert_sent = bool(result[1]) if result[1] is not None else False
                        else:
                            first_seen_time = current_time
                            alert_sent = False
                    else:
                        face_id = f"face_{next_face_id}"
                        next_face_id += 1
                        first_seen_time = current_time
                        alert_sent = False
                        
                        face_image_bytes = cv2.imencode('.jpg', face_roi)[1].tobytes()
                        
                        cursor.execute('''
                            INSERT OR IGNORE INTO faces (face_id, first_seen, last_seen, total_time, face_image)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (face_id, datetime.fromtimestamp(current_time), 
                              datetime.fromtimestamp(current_time), 0, face_image_bytes))
                        conn.commit()
                    
                    if face_id not in face_trackers:
                        face_trackers[face_id] = {
                            'first_seen': first_seen_time,
                            'last_seen': current_time,
                            'last_box': face_box,
                            'alert_sent': alert_sent
                        }
                    else:
                        face_trackers[face_id]['last_seen'] = current_time
                        face_trackers[face_id]['last_box'] = face_box
                        if 'alert_sent' in locals():
                            face_trackers[face_id]['alert_sent'] = alert_sent
                    
                    active_face_ids.add(face_id)
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
        text = f"{face_id} ({int(elapsed_time)}s)"
        frame = put_text_ru(frame, text, (x, max(0, y - 20)), 16, color, 2)
    
    person_count = len(face_trackers)
    text = f"Людей в кадре: {person_count}"
    frame = put_text_ru(frame, text, (10, 10), 24, (0, 255, 255), 2)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
