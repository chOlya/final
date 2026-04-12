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
        logging.info("Модель загружена")
    
    async def handle_connection(self, websocket, path):
        logging.info("Клиент подключился")
        
        # Накопление всех аудиоданных от всех чанков
        full_audio_buffer = bytearray()
        
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
                            logging.info(f"Получен EOF, всего данных: {len(full_audio_buffer)} байт")
                            
                            if len(full_audio_buffer) > 0:
                                # Транскрибируем всё накопленное аудио
                                result = await self.transcribe(full_audio_buffer)
                                if result:
                                    await websocket.send(json.dumps(result))
                                    logging.info(f"Отправлен результат: {len(result['result'])} слов")
                            else:
                                logging.warning("Нет аудиоданных для обработки")
                            
                            # Не закрываем соединение сразу, даем время на отправку
                            await asyncio.sleep(0.1)
                            break
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"Ошибка парсинга JSON: {e}")
                    except Exception as e:
                        logging.error(f"Ошибка обработки сообщения: {e}")
                else:
                    # Накопление аудиоданных
                    full_audio_buffer.extend(message)
                    logging.debug(f"Получено аудио: {len(message)} байт, всего: {len(full_audio_buffer)}")
                    
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
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            logging.info(f"Аудио для распознавания: {len(audio_float32)} сэмплов, {len(audio_float32)/self.sample_rate:.2f} сек")
            
            # Распознавание
            result = self.model.transcribe(
                audio_float32,
                language="ru",           # Русский язык
                word_timestamps=True,    # Таймстемпы слов
                fp16=False,              # Отключаем FP16 для совместимости
                temperature=0.0,         # Детерминированный вывод
                no_speech_threshold=0.6, # Порог тишины
                condition_on_previous_text=False  # Не ждем предыдущий текст
            )
            
            # Форматирование результата
            formatted = {
                "text": result["text"],
                "result": []
            }
            
            # Извлечение слов с таймстемпами
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    formatted["result"].append({
                        "word": word["word"],
                        "start": round(word["start"], 2),
                        "end": round(word["end"], 2),
                        "confidence": round(word.get("probability", 1.0), 3)
                    })
            
            logging.info(f"Распознано {len(formatted['result'])} слов")
            logging.info(f"Текст: {result['text'][:200]}...")
            
            return formatted
            
        except Exception as e:
            logging.error(f"Ошибка транскрипции: {e}", exc_info=True)
            return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v3"],
                       help="Размер модели Whisper")
    parser.add_argument("--host", default="localhost", help="Хост")
    parser.add_argument("--port", type=int, default=2701, help="Порт")
    args = parser.parse_args()
    
    server = WhisperServer(model_size=args.model)
    
    async with websockets.serve(
        server.handle_connection, 
        args.host, 
        args.port,
        ping_interval=None,  # Отключаем ping для стабильности
        close_timeout=10     # Увеличиваем таймаут закрытия
    ):
        logging.info(f"Whisper сервер запущен на ws://{args.host}:{args.port}")
        logging.info(f"Модель: {args.model}")
        logging.info("Сервер готов к работе")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())