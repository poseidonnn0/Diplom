import json
import sys
import io
import threading
import time
import logging
import secrets
import locale
from collections import deque
from datetime import datetime, timedelta
from threading import Lock, Thread
from copy import deepcopy
from threading import Event

import cv2
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    send_file, session, redirect, url_for, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from urllib.parse import quote_plus
from facenet_pytorch import MTCNN
import torch
from appTestCamera import CAMERA_INDEX

# Настройка кодировки
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
locale.setlocale(locale.LC_ALL, 'Russian')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.secret_key = 'd3b07384d113edec49eaa6238ad5ff00'

frame_available = Event()

# Настройки камеры
CAMERA_SETTINGS = {
    'max_retries': 3,
    'timeout': 5.0,
    'frame_cache_size': 3,
    'emergency_fallback': True
}

# Глобальный кеш последних кадров
frame_cache = deque(maxlen=CAMERA_SETTINGS['frame_cache_size'])

@app.before_request
def check_auth():
    if request.endpoint in ['admin_panel', 'handle_camera_update', 'handle_zone_update']:
        if not session.get('admin_auth'):
            return redirect(url_for('admin_login'))


# Настройка кодировки

ADMIN_USER = {
    "login": "admin",
    "password_hash": generate_password_hash("your_secure_password_here")
}

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Конфигурация PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'people_counter',
    'user': 'pc_user',
    'password': 'strong_password'
}

PERSISTENT_SETTINGS = {
    'camera_type': 'usb',
    'camera_url': '0',
    'zones': []
}
# Инициализация SQLAlchemy
encoded_password = quote_plus(DB_CONFIG['password'])
DSN = f"postgresql+psycopg2://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Конфигурация SQLAlchemy с NullPool для многопроцессорности
engine = create_engine(DSN, poolclass=NullPool)

# Константы
MAX_DAYS_PERIOD = 31
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Глобальные состояния с блокировками
camera_lock = Lock()
zones_lock = Lock()
current_cap = None
zones_config = []
last_counts = {}
frame_size = (1920, 1080)
model = YOLO("yolov10x.pt")

# После инициализации модели YOLO
mtcnn = MTCNN(
    keep_all=True,
    post_process=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Конфигурация зон
video_path = "videoAUD.mp4"

cap = cv2.VideoCapture(video_path) #Для работы с готовым видеорядом
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()



from concurrent.futures import ThreadPoolExecutor
import atexit

# Глобальные объекты
video_executor = ThreadPoolExecutor(max_workers=2)
camera_switch_lock = Lock()
current_stream = None

def release_camera():
    global current_cap
    with camera_lock:
        if current_cap and current_cap.isOpened():
            try:
                current_cap.release()
            except Exception as e:
                logging.error(f"Camera release error: {str(e)}")
            finally:
                current_cap = None

atexit.register(release_camera)


def load_persistent_settings():
    try:
        with engine.connect() as conn:
            result = conn.execute(text('SELECT * FROM admin_settings ORDER BY id DESC LIMIT 1'))
            settings = result.fetchone()

            if settings:
                PERSISTENT_SETTINGS.update({
                    'camera_type': settings.camera_type,
                    'camera_url': settings.camera_url,
                    'zones': json.loads(settings.zones) if settings.zones else []
                })
    except Exception as e:
        logging.error(f"Settings load error: {str(e)}")

def save_persistent_settings():
    try:
        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO admin_settings (camera_type, camera_url, zones, last_update)
                VALUES (:type, :url, :zones, NOW())
            '''), {
                'type': PERSISTENT_SETTINGS['camera_type'],
                'url': PERSISTENT_SETTINGS['camera_url'],
                'zones': json.dumps(PERSISTENT_SETTINGS['zones'])
            })
            conn.commit()
    except Exception as e:
        logging.error(f"Settings save error: {str(e)}")


def init_camera_config():
    global current_cap, zones_config, frame_size

    load_persistent_settings()

    with camera_lock:
        try:
            if PERSISTENT_SETTINGS['camera_type'] == 'ip':
                # Для IP-камер используйте CAP_FFMPEG
                current_cap = cv2.VideoCapture(PERSISTENT_SETTINGS['camera_url'], cv2.CAP_FFMPEG)
            else:
                # Для USB-камер явно укажите CAP_DSHOW
                current_cap = cv2.VideoCapture(int(PERSISTENT_SETTINGS['camera_url']), cv2.CAP_DSHOW)

            if not current_cap.isOpened():
                raise RuntimeError("Не удалось подключиться к камере")

            frame_size = (
                int(current_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        except Exception as e:

            logging.error(f"Ошибка инициализации камеры: {str(e)}")

            current_cap = cv2.VideoCapture(video_path)  # Фолбек на видеофайл

            frame_size = (width, height)


        # Инициализация зон
        with zones_lock:
            if PERSISTENT_SETTINGS.get('zones'):
                zones_config = PERSISTENT_SETTINGS['zones']
            else:
                # Дефолтные зоны
                zones_config = [
            {
                'name': 'Первая зона',
                'coords': (0, 0, frame_size[0] // 3, frame_size[1]),
                'max_capacity': 7,
                'color': '#2ecc71'
            },
            {
                'name': 'Вторая зона',
                'coords': (frame_size[0] // 3, 0, 2 * frame_size[0] // 3, frame_size[1]),
                'max_capacity': 9,
                'color': '#e67e22'
            },
            {
                'name': 'Третья зона',
                'coords': (2 * frame_size[0] // 3, 0, frame_size[0], frame_size[1]),
                'max_capacity': 10,
                'color': '#9b59b6'
            }
        ]






# Инициализация при старте
init_camera_config()



# Авторизация администратора
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if (request.form['username'] == ADMIN_USER['login'] and
            check_password_hash(ADMIN_USER['password_hash'], request.form['password'])):
            session['admin_auth'] = True
            return redirect(url_for('admin_panel'))
        return render_template('admin_login.html', error='Неверные данные')
    return render_template('admin_login.html')


@app.route('/admin/panel')
def admin_panel():
    if not session.get('admin_auth'):
        return redirect(url_for('admin_login'))

    return render_template(
        'admin_dashboard.html',
        zones=zones_config
    )


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_auth', None)
    session.pop('_csrf_token', None)  # Очистка CSRF-токена
    return redirect(url_for('index'))

@app.after_request
def add_cors_headers(response):
    # Разрешаем запросы с любых доменов
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-CSRFToken'
    return response


@app.route('/video_feed')
def video_feed():
    def generate():
        gen = None
        try:
            gen = video_stream_generator()
            while True:
                frame = next(gen)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            if gen:
                gen.close()
            raise
        except Exception as e:
            logging.error(f"Client connection error: {str(e)}")
        finally:
            if gen:
                gen.close()
            release_camera()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Инициализация камеры по умолчанию
def init_camera():
    global current_cap
    with camera_lock:
        if current_cap is None or not current_cap.isOpened():
            current_cap = cv2.VideoCapture(0)
    return current_cap


def process_frame(frame):
    display_frame = frame.copy()
    boxes = []
    counts = {zone['name']: 0 for zone in zones_config}
    last_boxes = []

    # Детекция объектов с помощью YOLO
    results = model.track(frame, classes=[0], persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        with zones_lock:
            current_zones = deepcopy(zones_config)

        for idx, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            last_boxes.append((x1, y1, x2, y2, conf))

            # Проверка попадания в зоны
            for zone in current_zones:
                z_x1, z_y1, z_x2, z_y2 = zone['coords']
                if z_x1 <= x_center <= z_x2 and z_y1 <= y_center <= z_y2:
                    counts[zone['name']] += 1
                    break

            # Отрисовка боксов
            color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame,
                        f"ID:{track_id} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
            cv2.circle(display_frame, (x_center, y_center), 6, (0, 0, 255), -1)

    # Распознавание лиц с помощью MTCNN
    boxes_faces, _ = mtcnn.detect(frame)
    if boxes_faces is not None:
        for box in boxes_faces:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Отрисовка зон и счетчиков
    with zones_lock:
        current_zones = deepcopy(zones_config)

    for zone in current_zones:
        x1, y1, x2, y2 = zone['coords']
        color = tuple(int(zone['color'][i:i + 2], 16) for i in (1, 3, 5))

        # Рамка зоны
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)

        # Фон для текста
        text = f"{zone['name']}: {counts[zone['name']]}/{zone['max_capacity']}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(display_frame,
                      (x1, y1 - text_height - 20),
                      (x1 + text_width + 20, y1),
                      color, -1)

        # Текст счетчика
        cv2.putText(display_frame, text,
                    (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3)

    # Сохранение данных в БД
    save_to_database(counts)

    return display_frame

def save_to_database(counts):
    try:
        with engine.connect() as conn:
            timestamp = datetime.now()
            for zone_name, count in counts.items():
                conn.execute(
                    text('''
                        INSERT INTO zone_counts (timestamp, zone_name, count)
                        VALUES (:ts, :zone, :cnt)
                    '''),
                    {'ts': timestamp, 'zone': zone_name, 'cnt': count}
                )
            conn.commit()
    except Exception as e:
        logging.error(f"Ошибка сохранения в БД: {str(e)}")


# Обновленная функция generate_frames()
def generate_frames():
    fps_limiter = 20


    while True:
        try:
            with camera_lock:
                if not current_cap or not current_cap.isOpened():
                    init_camera_config()
                    continue

                # Быстрый захват кадра без длительных операций
                ret, frame = current_cap.read()
                if not ret:
                    raise RuntimeError("Ошибка захвата кадра")

            # Обработка вне блокировки
            processed_frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            logging.error(f"Ошибка генерации: {str(e)}")
            time.sleep(1)


@app.route('/admin/update_camera', methods=['POST'])
def handle_camera_update():
    global current_cap, frame_size

    if not session.get('admin_auth'):
        return jsonify({"status": "unauthorized"}), 403

    data = request.get_json()
    new_cap = None
    success = False

    try:
        # Освобождение предыдущей камеры
        release_thread = threading.Thread(target=release_camera)
        release_thread.start()
        release_thread.join(timeout=2)

        # Инициализация новой камеры
        if data['type'] == 'ip':
            new_cap = cv2.VideoCapture(data['url'], cv2.CAP_FFMPEG)
            new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            new_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        else:
            new_cap = cv2.VideoCapture(int(data['url']), cv2.CAP_DSHOW)

        # Проверка подключения
        start_time = time.time()
        while not new_cap.isOpened() and (time.time() - start_time) < 5:
            time.sleep(0.1)

        if not new_cap.isOpened():
            raise RuntimeError("Camera connection failed")

        # Тестовый кадр
        ret, _ = new_cap.read()
        if not ret:
            raise RuntimeError("No video signal")

        # Атомарное обновление
        with camera_lock:
            current_cap = new_cap
            frame_size = (
                int(current_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

        # Сохранение настроек
        PERSISTENT_SETTINGS.update({
            'camera_type': data['type'],
            'camera_url': data['url']
        })
        save_persistent_settings()
        success = True

        return jsonify({
            "status": "ok",
            "message": "Camera switched successfully",
            "resolution": f"{int(frame_size[0])}x{int(frame_size[1])}"
        })

    except Exception as e:
        if new_cap:
            new_cap.release()
        logging.error(f"Camera update failed: {str(e)}")

        # Восстановление предыдущих настроек при ошибке
        if success:
            init_camera_config()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


def video_stream_generator():
    retry_count = 0
    MAX_RETRIES = 3
    while True:
        try:
            with camera_switch_lock:
                if not current_cap or not current_cap.isOpened():
                    init_camera_config()

                ret, frame = current_cap.read()
                if not ret:
                    retry_count += 1
                    if retry_count > MAX_RETRIES:
                        init_camera_config()
                        retry_count = 0
                    continue

                retry_count = 0
                processed_frame = process_frame(cv2.resize(frame, (1280, 720)))
                _, buffer = cv2.imencode('.jpg', processed_frame)
                yield buffer.tobytes()
                time.sleep(0.033)  # Добавить задержку для 30 FPS

        except Exception as e:
            logging.error(f"Stream error: {str(e)}")
            time.sleep(1)


@app.route('/admin/update_zone', methods=['POST'])
def handle_zone_update():
    if not session.get('admin_auth'):
        return jsonify({"status": "unauthorized"}), 403

    try:
        data = request.json
        # Валидация координат
        if len(data['coords']) != 4:
            raise ValueError("Некорректные координаты зоны")

        with zones_lock:
            zones_config[data['index']].update({
                'name': data['name'],
                'coords': tuple(map(int, data['coords'])),
                'max_capacity': int(data['max_capacity']),
                'color': data['color']
            })
            # Сохраняем в постоянные настройки
            PERSISTENT_SETTINGS['zones'] = zones_config
            save_persistent_settings()

        return jsonify({"status": "ok"})

    except Exception as e:
        logging.error(f"Ошибка обновления зоны: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

# Основные маршруты
@app.route('/')
def index():
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, DATE_FORMAT)
        except:
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

    try:
        start_str = request.args.get('start')
        end_str = request.args.get('end')
        error = None

        if start_str and end_str:
            start_dt = parse_date(start_str)
            end_dt = parse_date(end_str)
            if start_dt > end_dt:
                error = "Начальная дата должна быть раньше конечной"
            elif (end_dt - start_dt).days > MAX_DAYS_PERIOD:
                error = f"Максимальный период: {MAX_DAYS_PERIOD} дней"
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(hours=24)

        if error:
            return render_template('index.html', error=error)

        with engine.connect() as conn:
            result = conn.execute(text('''
                WITH latest AS (
                    SELECT zone_name, count, timestamp,
                           ROW_NUMBER() OVER (PARTITION BY zone_name ORDER BY timestamp DESC) as rn
                    FROM zone_counts
                    WHERE timestamp BETWEEN :start AND :end
                ),
                peaks AS (
                    SELECT zone_name, MAX(count) as peak
                    FROM zone_counts
                    WHERE timestamp BETWEEN :start AND :end
                    GROUP BY zone_name
                )
                SELECT l.zone_name, l.count as current, l.timestamp, p.peak
                FROM latest l
                LEFT JOIN peaks p ON l.zone_name = p.zone_name
                WHERE rn = 1
            '''), {'start': start_dt, 'end': end_dt})

            zones_db = {row.zone_name: row for row in result}

        zones_data = []
        total_count = 0
        last_update = None

        with zones_lock:
            current_zones = deepcopy(zones_config)

        for zone in current_zones:
            data = zones_db.get(zone['name'])
            if data:
                current = data.current
                peak = data.peak
                timestamp = data.timestamp
                if last_update is None or timestamp > last_update:
                    last_update = timestamp
            else:
                current = 0
                peak = 0
                timestamp = None

            zones_data.append({
                **zone,
                'count': current,
                'peak_today': peak,
                'timestamp': timestamp
            })
            total_count += current

        return render_template('index.html',
                            zones=zones_data,
                            total_count=total_count,
                            last_update=last_update,
                            start_date=start_dt.strftime(DATE_FORMAT),
                            end_date=end_dt.strftime(DATE_FORMAT),
                            error=error)

    except Exception as e:
        logging.error(f"Ошибка: {str(e)}")
        return render_template('index.html',
                            error=str(e),
                            start_date='',
                            end_date='')


@app.route('/download', methods=['GET'])
def download_data():
    try:
        logging.debug("=" * 50)
        logging.debug("Starting download request processing")
        logging.debug(f"Raw request args: {request.args}")

        # Парсинг параметров
        start_str = request.args.get('start', '')
        end_str = request.args.get('end', '')
        format = request.args.get('format', 'xlsx')
        logging.debug(f"Received params: start='{start_str}' end='{end_str}' format='{format}'")

        # Валидация наличия параметров
        if not start_str or not end_str:
            logging.error("Missing start/end parameters")
            return jsonify({"error": "Не указаны start/end параметры"}), 400

        # Парсинг дат
        def parse_date(date_str):
            try:
                # Пробуем парсить с секундами
                return datetime.strptime(date_str, DATE_FORMAT)
            except ValueError:
                try:
                    # Пробуем парсить без секундов (формат datetime-local)
                    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
                except Exception as e:
                    logging.error(f"Failed to parse date '{date_str}': {str(e)}")
                    raise

        try:
            logging.debug("Attempting to parse dates...")
            start_dt = parse_date(start_str)
            end_dt = parse_date(end_str)
            logging.debug(f"Parsed dates: start={start_dt} end={end_dt}")
        except Exception as e:
            logging.error(f"Date parsing failed: {str(e)}", exc_info=True)
            return jsonify({"error": f"Ошибка формата даты: {str(e)}"}), 400

        # Валидация периода
        if (end_dt - start_dt).days > MAX_DAYS_PERIOD:
            logging.error(f"Period too long: {end_dt - start_dt} days")
            return jsonify({"error": f"Максимальный период: {MAX_DAYS_PERIOD} дней"}), 400

        # SQL запрос
        try:
            logging.debug("Executing database query...")
            with engine.connect() as conn:
                query = text('''
                    SELECT 
                        TO_CHAR(timestamp, 'YYYY-MM-DD HH24:MI:SS') as timestamp,
                        zone_name,
                        count
                    FROM zone_counts
                    WHERE timestamp BETWEEN :start AND :end
                    ORDER BY timestamp DESC
                ''')
                logging.debug(f"SQL query: {query}")
                logging.debug(f"Query params: start={start_dt} end={end_dt}")

                df = pd.read_sql_query(query, conn, params={'start': start_dt, 'end': end_dt})
                logging.debug(f"Retrieved {len(df)} rows from database")

                if df.empty:
                    logging.warning("No data found for specified period")
                    return jsonify({"error": "Данные не найдены"}), 404

        except Exception as e:
            logging.error("Database error occurred:", exc_info=True)
            return jsonify({"error": f"Ошибка базы данных: {str(e)}"}), 500

        # Генерация файла
        try:
            logging.debug(f"Generating {format.upper()} file...")
            output = io.BytesIO()

            if format == 'csv':
                df.to_csv(output, index=False)
                mimetype = 'text/csv'
            else:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

            output.seek(0)
            logging.debug("File generated successfully")

        except Exception as e:
            logging.error("File generation failed:", exc_info=True)
            return jsonify({"error": f"Ошибка генерации файла: {str(e)}"}), 500

        output.seek(0)
        return send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f'people_data.{format}'
        )

    except Exception as e:
        logging.critical("Unhandled exception in download endpoint:", exc_info=True)
        return jsonify({"error": "Критическая ошибка сервера"}), 500


def video_processing():
    global current_cap
    while True:
        with camera_lock:
            if current_cap is None:
                current_cap = cv2.VideoCapture(CAMERA_INDEX)
            cap = current_cap

        ret, frame = cap.read()
        if not ret:
            continue

    import time
    logging.info("Starting video processing thread")

    try:
        local_engine = create_engine(DSN, poolclass=NullPool)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error("Error opening video file")
            return

        last_save_time = datetime.now()
        frame_interval = 1 / 30  # 30 FPS
        last_counts = {zone['name']: 0 for zone in zones_config}
        last_boxes = []

        # Настройка полноэкранного режима
        cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("People Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("People Counter", 1920, 1080)

        while True:
            start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            display_frame = frame.copy()
            # Отрисовка зон
            for zone in zones_config:
                x1, y1, x2, y2 = zone['coords']
                color = tuple(int(zone['color'][i:i + 2], 16) for i in (1, 3, 5))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Добавлен фон для текста
                zone_text = f"{zone['name']}: {last_counts[zone['name']]}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.9
                thickness = 2
                text_color = (0, 0, 0)
                bg_color = (255, 255, 255)  # Белый фон
                padding = 3

                # Рассчитываем размеры текста
                (text_width, text_height), baseline = cv2.getTextSize(zone_text, font, scale, thickness)

                # Позиция текста (базовая линия)
                x_text = x1 + 10
                y_text = y1 + 30

                # Координаты фона
                bg_x1 = x_text - padding
                bg_y1 = y_text - text_height - padding
                bg_x2 = x_text + text_width + padding
                bg_y2 = y_text + baseline + padding

                # Рисуем фон
                cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)

                # Рисуем текст поверх фона
                cv2.putText(display_frame,
                            zone_text,  # Используем новое имя
                            (x_text, y_text),
                            font,
                            scale,
                            text_color,
                            thickness)

            # Отрисовка объектов
            for box_info in last_boxes:
                x1_box, y1_box, x2_box, y2_box, conf = box_info

                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)

                cv2.rectangle(display_frame, (x1_box, y1_box), (x2_box, y2_box), (0, 0, 0), 2)

                # Добавляем фон ТОЛЬКО для текста accuracy
                accuracy_text = f"accuracy: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thickness = 2
                text_size = cv2.getTextSize(accuracy_text, font, scale, thickness)[0]

                # Координаты текста (как были изначально)
                text_x = x1_box + 5
                text_y = y1_box - 10

                # Параметры фона
                bg_color = (255, 255, 255)  # Черный фон
                padding = 3
                bg_x1 = text_x - padding
                bg_y1 = text_y - text_size[1] - padding
                bg_x2 = text_x + text_size[0] + padding
                bg_y2 = text_y + padding

                # Рисуем фон
                cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)

                # Оригинальный текст (без изменений в параметрах)
                cv2.putText(display_frame,
                            accuracy_text,
                            (text_x, text_y),
                            font,
                            scale,
                            (0, 0, 0),
                            thickness)

                x_center = (x1_box + x2_box) // 2
                y_center = (y1_box + y2_box) // 2
                cv2.circle(display_frame, (x_center, y_center), 6, (0, 0, 255), -1)

            current_time = datetime.now()
            if (current_time - last_save_time).total_seconds() >= 3:
                try:
                    results = model(frame, classes=[0], verbose=False)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    counts = {zone['name']: 0 for zone in zones_config}
                    last_boxes.clear()

                    for idx, box in enumerate(boxes):
                        x1_box, y1_box, x2_box, y2_box = map(int, box)
                        conf = confidences[idx]
                        last_boxes.append((x1_box, y1_box, x2_box, y2_box, conf))

                        x_center = (x1_box + x2_box) // 2
                        y_center = (y1_box + y2_box) // 2
                        for zone in zones_config:
                            z_x1, z_y1, z_x2, z_y2 = zone['coords']
                            if z_x1 <= x_center <= z_x2 and z_y1 <= y_center <= z_y2:
                                counts[zone['name']] += 1
                                break

                    last_counts.update(counts)

                    with local_engine.connect() as conn:
                        for zone_name, count in counts.items():
                            conn.execute(
                                text('INSERT INTO zone_counts (timestamp, zone_name, count) VALUES (:ts, :zone, :cnt)'),
                                {'ts': current_time, 'zone': zone_name, 'cnt': count}
                            )
                            conn.commit()
                        logging.info(f"Data saved at {current_time}")

                    last_save_time = current_time

                except Exception as e:
                    logging.error(f"Processing error: {str(e)}", exc_info=True)

            cv2.imshow("People Counter", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logging.error(f"Video processing fatal error: {str(e)}", exc_info=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        local_engine.dispose() 


@app.route('/api/current_stats')
def get_current_stats(current_counts=None):
    with zones_lock:
        zones = deepcopy(zones_config)

    # Добавьте актуальные данные из детектора
    return jsonify({
        "zones": [
            {
                "name": z['name'],
                "count": current_counts[z['name']],
                "max_capacity": z['max_capacity']
            } for z in zones
        ]
    })

def create_tables():
    try:
        with engine.connect() as conn:

            conn.execute(text('SELECT 1'))

            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS zone_counts (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    zone_name VARCHAR(50) NOT NULL,
                    count INTEGER NOT NULL
                )
            '''))
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS admin_settings (
                    id SERIAL PRIMARY KEY,
                    camera_type VARCHAR(20),
                    camera_url TEXT,
                    last_update TIMESTAMP
                )
            '''))  # Новая таблица для настроек
            conn.commit()
    except Exception as e:
        logging.error(f"Database init error: {str(e)}")
        exit(1)


@app.route('/hourly_data')
def get_hourly_data():
    try:
        with engine.connect() as conn:
            result = conn.execute(text('''
                SELECT 
                    zone_name,
                    CASE 
                        WHEN EXTRACT(DOW FROM timestamp) = 0 THEN 6  -- Воскресенье -> 6
                        ELSE EXTRACT(DOW FROM timestamp) - 1         -- Пн-Сб -> 0-5
                    END AS day_of_week,
                    EXTRACT(HOUR FROM timestamp) AS hour,
                    AVG(count) AS avg_count  
                FROM zone_counts
                GROUP BY zone_name, day_of_week, hour
                ORDER BY zone_name, day_of_week, hour
            '''))

            data = {}
            for row in result:
                zone = row.zone_name
                day = int(row.day_of_week)
                hour = int(row.hour)
                avg = float(row.avg_count)  # Явное преобразование к float

                if zone not in data:
                    data[zone] = {}
                if day not in data[zone]:
                    data[zone][day] = {}

                data[zone][day][hour] = avg

            return jsonify({
                'days': ['Monday', 'Tuesday', 'Wednesday',
                        'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'zones': data
            })

    except Exception as e:
        logging.error(f"Hourly data error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    create_tables()

    from waitress import serve
    from multiprocessing import Process

    video_process = Process(target=video_processing)
    video_process.daemon = True
    video_process.start()
    # Ограничение ресурсов
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    # Запуск сервера
    serve(
        app,
        host='0.0.0.0',
        port=5000,
        threads=8,
        connection_limit=100,
        channel_timeout=30
    )