# 🎓 Дипломный проект: Система анализа посещаемости коворкингов

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.2-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![YOLOv10](https://img.shields.io/badge/YOLOv10-ultralytics-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-lightgrey)

**Умная система мониторинга загруженности пространств** с компьютерным зрением, аналитикой в реальном времени и веб-интерфейсом для управления.

## 🚀 Особенности
- **Точная детекция людей**: YOLOv10 с точностью до 92% и скоростью 60 FPS
- **Гибкие зоны мониторинга**: динамическая настройка областей анализа
- **Аналитика в реальном времени**: графики, тепловые карты, экспорт отчетов
- **Мультикамерная поддержка**: работа с USB/IP-камерами и RTSP-потоками
- **Отказоустойчивость**: автоматический переход на резервное видео при сбоях


## 📚 Оглавление
1. [Технологии](#-технологии)
2. [Установка](#-установка)
<<<<<<< Updated upstream
3. [Использование](#-использование)
4. [Архитектура](#-архитектура)
5. [Документация](#-документация)
6. [Лицензия](#-лицензия)
=======

>>>>>>> Stashed changes

## 🛠 Технологии
- **Backend**: 
  ![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)
  ![Flask](https://img.shields.io/badge/Flask-2.3.2-000000?logo=flask)
- **Компьютерное зрение**: 
  ![YOLOv10](https://img.shields.io/badge/YOLOv10-ultralytics-red)
  ![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv)
- **База данных**: 
  ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql)
- **Frontend**: 
  ![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?logo=chart.js)
  ![Jinja2](https://img.shields.io/badge/Jinja2-3.1.2-b41717)

## 💻 Установка
```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/coworking-analytics.git
cd coworking-analytics

# Установить зависимости
pip install -r requirements.txt

# Настройка БД (заполнить .env по образцу .env.example)
createdb people_counter
python -c "from appDodelan import create_tables; create_tables()"

<<<<<<< Updated upstream
graph TD
    A[Камера] --> B[OpenCV]
    B --> C[YOLOv10]
    C --> D[Обработка зон]
    D --> E[PostgreSQL]
    E --> F[Flask API]
    F --> G[Веб-интерфейс]
    G --> H[Пользователь]
=======
>>>>>>> Stashed changes
