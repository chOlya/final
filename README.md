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
cd backend
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
.venv\Scripts\activate   # Windows
pip install websockets==9.1 vosk openai-whisper transformers torch torchaudio soundfile
```

### Запуск Vosk сервера

```bash
python vosk_server.py ../models/vosk-model-ru-0.42 --host localhost --port 2700
```

### Запуск Whisper сервера

```bash
python whisper_server.py --model medium --host localhost --port 2701
```

### Запуск Wav2Vec2 сервера

```bash
python wav2vec2_server.py --host localhost --port 2702
```

### Запуск frontend

В отдельном терминале
```bash
cd frontend
npm install
npm run dev
```

Проект запустится на http://localhost:5173/
