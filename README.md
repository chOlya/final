# final

### Требования

- Python 3.9+
- Node.js 18+

### Скачивание модели Vosk

Создание папки для моделей
```bash
mkdir -p models
```

Скачивание русской модели (рекомендуется для качества)
```bash
wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
unzip vosk-model-ru-0.42.zip -d models/
```

Лёгкая модель для быстрой работы
```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip -d models/
```

### Установка

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.venv\Scripts\activate   # Windows
```

### Запуск Vosk сервера

```bash
cd backend
pip install websockets==9.1 vosk   
python vosk_server.py ../models/vosk-model-ru-0.42
```

### Запуск frontend

В отдельном терминале из папки vosk-react
```bash
npm install
npm run dev
```

Проект запустится на http://localhost:5173/
