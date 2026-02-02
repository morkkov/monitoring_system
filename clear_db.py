import sqlite3
import os

db_path = 'faces.db'

if not os.path.exists(db_path):
    print(f"База данных {db_path} не найдена")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM faces')
count_before = cursor.fetchone()[0]
print(f"Записей в базе до очистки: {count_before}")

confirm = input("Вы уверены, что хотите очистить базу данных? (yes/no): ")

if confirm.lower() == 'yes':
    cursor.execute('DELETE FROM faces')
    conn.commit()
    
    cursor.execute('SELECT COUNT(*) FROM faces')
    count_after = cursor.fetchone()[0]
    print(f"База данных очищена. Записей осталось: {count_after}")
else:
    print("Очистка отменена")

conn.close()
