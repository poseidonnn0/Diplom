import os
import time

from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import threading
import io
import logging
import xlsxwriter
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from sqlalchemy.pool import NullPool
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.secret_key = 'd3b07384d113edec49eaa6238ad5ff00'

# Конфигурация PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'people_counter',
    'user': 'pc_user',
    'password': 'strong_password'
}

# Инициализация SQLAlchemy
encoded_password = quote_plus(DB_CONFIG['password'])
DSN = f"postgresql+psycopg2://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Конфигурация SQLAlchemy с NullPool для многопроцессорности
engine = create_engine(DSN, poolclass=NullPool)

# Константы
MAX_DAYS_PERIOD = 31
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'



# Настройки FFmpeg для HLS
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "1000"
os.environ["OPENCV_FFMPEG_READ_ATTEMPT_STEP"] = "30000"
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "4"  # Для многопоточного декодирования

# Конфигурация камеры
HLS_URL = "https://s2.moidom-stream.ru/s/public/0000001657.m3u8"
RECONNECT_DELAY = 10  # Увеличенный таймаут для HLS

# Инициализация параметров видео
cap = cv2.VideoCapture(HLS_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    logging.error("Ошибка инициализации видео. Проверьте поддержку HLS в FFmpeg")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

zones_config = [
    {
        'name': 'First zone',
        'coords': (0, 0, width // 3, height),
        'max_capacity': 7,
        'color': '#2ecc71'
    },
    {
        'name': 'Second zone',
        'coords': (width // 3, 0, 2 * width // 3, height),
        'max_capacity': 9,
        'color': '#e67e22'
    },
    {
        'name': 'Third zone',
        'coords': (2 * width // 3, 0, width, height),
        'max_capacity': 10,
        'color': '#9b59b6'
    },
]

# Инициализация модели
model = YOLO("yolov10x.pt")


@app.route('/admin/update_camera', methods=['POST'])
def handle_camera_update():
    if not session.get('admin_auth'):
        return jsonify({"status": "unauthorized"}), 403

    try:
        data = request.json
        global cap

        # Правильное освобождение ресурсов
        if 'cap' in globals() and cap.isOpened():
            cap.release()
            time.sleep(1)  # Пауза для завершения операций

        # Инициализация новой камеры
        if data['type'] == 'ip':
            cap = cv2.VideoCapture(data['url'], cv2.CAP_FFMPEG)
        elif data['type'] == 'usb':
            cap = cv2.VideoCapture(int(data['url']))
        else:
            raise ValueError("Invalid camera type")

        # Проверка подключения
        if not cap.isOpened():
            raise RuntimeError("Camera initialization failed")

        return jsonify({"status": "success", "message": "Камера успешно подключена"})

    except Exception as e:
        logging.error(f"Camera update error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/admin/update_zone', methods=['POST'])
def handle_zone_update():
    if not session.get('admin_auth'):
        return jsonify({"status": "unauthorized"}), 403

    try:
        data = request.json
        index = data['index']

        # Валидация координат
        if len(data['coords']) != 4 or any(not isinstance(c, int) for c in data['coords']):
            raise ValueError("Invalid coordinates format")

        zones_config[index].update({
            'name': data['name'],
            'max_capacity': int(data['max_capacity']),
            'color': data['color'],
            'coords': data['coords']
        })

        # Обновление разрешения
        global width, height
        width = max(z['coords'][2] for z in zones_config)
        height = max(z['coords'][3] for z in zones_config)

        return jsonify({"status": "success"})

    except Exception as e:
        logging.error(f"Zone update error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.after_request
def add_cors_headers(response):
    # Разрешаем запросы с любых доменов
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


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
                error = "Start date must be before end date"
            elif (end_dt - start_dt).days > MAX_DAYS_PERIOD:
                error = f"Maximum period is {MAX_DAYS_PERIOD} days"
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(hours=24)

        if error:
            return render_template('index.html', error=error)

        with engine.connect() as conn:
            # Получаем последние значения и пики
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

        for zone in zones_config:
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
        print(zones_data)

        return render_template('index.html',
                               zones=zones_data,
                               total_count=total_count,
                               last_update=last_update,
                               start_date=start_dt.strftime(DATE_FORMAT),
                               end_date=end_dt.strftime(DATE_FORMAT),
                               error=error)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
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
    import time
    logging.info("Starting video processing thread")

    try:
        local_engine = create_engine(DSN, poolclass=NullPool)
        cap = cv2.VideoCapture(HLS_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # 60 секунд на открытие
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)  # 30 секунд на чтение

        if not cap.isOpened():
            logging.error("Error opening video file")
            return

        logging.info("Камера подключена")

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

def create_tables():
    try:
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS zone_counts (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    zone_name VARCHAR(50) NOT NULL,
                    count INTEGER NOT NULL
                )
            '''))
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

    serve(app, host='0.0.0.0', port=5000)