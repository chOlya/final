#!/usr/bin/env python3
"""
Whisper WebSocket сервер
"""

import asyncio
import json
import argparse
import logging
import numpy as np
import whisper
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WhisperServer:
    def __init__(self, model_size="medium"):
        logging.info(f"Загрузка модели {model_size}...")
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000
        logging.info(f"Модель {model_size} загружена")
    
    async def handle_connection(self, websocket, path):
        logging.info("Клиент подключился")
        audio_buffer = bytearray()
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        
                        # Обработка конфигурации
                        if 'config' in data:
                            logging.info(f"Получена конфигурация: {data['config']}")
                        
                        # Обработка EOF - конец передачи
                        if 'eof' in data and data['eof'] == 1:
                            logging.info(f"Получен EOF, всего данных: {len(audio_buffer)} байт")
                            
                            if len(audio_buffer) > 0:
                                # Транскрибируем всё аудио
                                result = await self.transcribe(audio_buffer)
                                if result:
                                    await websocket.send(json.dumps(result))
                                    logging.info(f"Отправлен результат: {len(result.get('result', []))} слов")
                                else:
                                    logging.warning("Пустой результат транскрипции")
                            else:
                                logging.warning("Нет аудиоданных для обработки")
                            
                            break
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"Ошибка парсинга JSON: {e}")
                else:
                    # Накопление аудиоданных
                    audio_buffer.extend(message)
                    if len(audio_buffer) % (16000 * 2 * 10) == 0:  # Каждые ~10 секунд
                        logging.debug(f"Получено аудио: {len(audio_buffer) / (16000 * 2):.1f} сек")
                    
        except websockets.exceptions.ConnectionClosed:
            logging.info("Клиент отключился")
        except Exception as e:
            logging.error(f"Ошибка в обработчике: {e}", exc_info=True)
        finally:
            logging.info("Соединение закрыто")
    
    async def transcribe(self, audio_buffer):
        try:
            # Конвертация 16-bit PCM в float32
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            
            if len(audio_int16) == 0:
                logging.warning("Пустой аудиобуфер")
                return None
                
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Легкая нормализация громкости
            max_val = np.max(np.abs(audio_float32))
            if max_val > 0 and max_val < 0.1:
                audio_float32 = audio_float32 / max_val * 0.95
            
            duration = len(audio_float32) / self.sample_rate
            logging.info(f"Транскрипция аудио: {len(audio_float32)} сэмплов, {duration:.2f} сек")
            
            # Транскрипция с оптимальными параметрами
            result = self.model.transcribe(
                audio_float32,
                language="ru",
                word_timestamps=True,
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.4,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.0,
                logprob_threshold=-1.0,
                initial_prompt="Раз, два, три, четыре, пять, шесть, семь, восемь, девять, десять."
            )
            
            # Форматирование результата
            formatted = {
                "text": result["text"].strip(),
                "result": []
            }
            
            # Извлечение слов с таймстемпами
            if "segments" in result:
                for segment in result["segments"]:
                    if "words" in segment:
                        for word in segment["words"]:
                            formatted["result"].append({
                                "word": word["word"].strip(),
                                "start": round(word["start"], 3),
                                "end": round(word["end"], 3),
                                "confidence": round(word.get("probability", 0.85), 3)
                            })
            
            # Если нет слов с таймстемпами, но есть текст
            if len(formatted["result"]) == 0 and result["text"]:
                words = result["text"].strip().split()
                if len(words) > 0 and duration > 0:
                    word_duration = duration / len(words)
                    for i, word in enumerate(words):
                        formatted["result"].append({
                            "word": word,
                            "start": round(i * word_duration, 3),
                            "end": round((i + 1) * word_duration, 3),
                            "confidence": 0.85
                        })
            
            logging.info(f"Распознано {len(formatted['result'])} слов")
            if formatted["text"]:
                logging.info(f"Текст: {formatted['text'][:200]}...")
            
            return formatted
            
        except Exception as e:
            logging.error(f"Ошибка транскрипции: {e}", exc_info=True)
            return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v3"],
                       help="Размер модели Whisper")
    parser.add_argument("--host", default="0.0.0.0", help="Хост")
    parser.add_argument("--port", type=int, default=2701, help="Порт")
    args = parser.parse_args()
    
    server = WhisperServer(model_size=args.model)
    
    async with websockets.serve(
        server.handle_connection, 
        args.host, 
        args.port,
        ping_interval=20,
        ping_timeout=60,
        close_timeout=10,
        max_size=50 * 1024 * 1024  # 50MB для больших аудиофайлов
    ):
        logging.info(f"Whisper сервер запущен на ws://{args.host}:{args.port}")
        logging.info(f"Модель: {args.model}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())