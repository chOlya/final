#!/usr/bin/env python3
"""
Vosk WebSocket сервер для речевого тренажёра
"""

import asyncio
import json
import argparse
import logging
from vosk import Model, KaldiRecognizer, SetLogLevel
import websockets

SetLogLevel(-1)

model = None

async def handle_connection(websocket, path):
    """Обработчик WebSocket соединения"""
    global model
    
    logging.info("Клиент подключился")
    
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    
    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    # Обработка конфигурации
                    if 'config' in data:
                        logging.info("Получена конфигурация")
                    # Обработка EOF
                    if 'eof' in data and data['eof'] == 1:
                        logging.info("Получен EOF, отправка финального результата...")
                        final = json.loads(rec.FinalResult())
                        if 'result' in final and final['result']:
                            try:
                                await websocket.send(json.dumps(final))
                                logging.info(f"Финальный результат отправлен: {final.get('text', '')[:50]}...")
                            except:
                                pass
                except json.JSONDecodeError:
                    pass
            else:
                # Аудиоданные
                if rec.AcceptWaveform(message):
                    result = json.loads(rec.Result())
                    if 'result' in result and result['result']:
                        try:
                            await websocket.send(json.dumps(result))
                            if len(result['result']) % 10 == 0:
                                logging.info(f"Распознано слов: {len(result['result'])}")
                        except:
                            pass
    
    except websockets.exceptions.ConnectionClosed:
        logging.info("Клиент отключился (ожидаемое поведение)")
    except Exception as e:
        logging.error(f"Ошибка: {e}")
    finally:
        # Не отправляем результат, если соединение уже закрыто
        logging.info("Обработчик завершён")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Путь к модели Vosk")
    parser.add_argument("--host", default="0.0.0.0", help="Хост")
    parser.add_argument("--port", type=int, default=2700, help="Порт")
    args = parser.parse_args()
    
    global model
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Загрузка модели из {args.model_path}...")
    model = Model(args.model_path)
    logging.info("Модель загружена")
    
    async with websockets.serve(handle_connection, args.host, args.port):
        logging.info(f"Сервер запущен на ws://{args.host}:{args.port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())