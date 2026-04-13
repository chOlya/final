#!/usr/bin/env python3
"""
Wav2Vec2 WebSocket сервер для русского языка
"""

import asyncio
import json
import argparse
import logging
import numpy as np
import torch
import websockets
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Wav2Vec2Server:
    def __init__(self, model_name="bond005/wav2vec2-large-ru-golos", use_gpu=False):
        logging.info(f"Загрузка модели Wav2Vec2: {model_name}...")
        
        # Определяем устройство
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("Используется GPU CUDA")
        else:
            self.device = torch.device("cpu")
            logging.info("Используется CPU")
        
        # Загрузка модели и процессора
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        self.sample_rate = 16000
        self.model.eval()
        
        logging.info(f"Модель загружена на {self.device}")
    
    async def handle_connection(self, websocket, path):
        logging.info("Клиент подключился")
        full_audio_buffer = bytearray()
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        if 'config' in data:
                            logging.info(f"Конфигурация: {data['config']}")
                        if 'eof' in data and data['eof'] == 1:
                            logging.info(f"Получен EOF, всего: {len(full_audio_buffer)} байт")
                            if len(full_audio_buffer) > 0:
                                result = await self.transcribe(full_audio_buffer)
                                if result:
                                    await websocket.send(json.dumps(result))
                                    logging.info(f"Отправлено {len(result['result'])} слов")
                            break
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON ошибка: {e}")
                else:
                    full_audio_buffer.extend(message)
                    
        except websockets.exceptions.ConnectionClosed:
            logging.info("Клиент отключился")
        except Exception as e:
            logging.error(f"Ошибка: {e}", exc_info=True)
    
    async def transcribe(self, audio_buffer):
        try:
            # Конвертация 16-bit PCM в float32
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            logging.info(f"Аудио: {len(audio_float32)/self.sample_rate:.2f} сек")
            
            # Подготовка входных данных
            input_values = self.processor(
                audio_float32, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_values.to(self.device)
            
            # Инференс
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Декодирование
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Получение слов с таймстемпами
            words = transcription.split()
            duration = len(audio_float32) / self.sample_rate
            step = duration / len(words) if words else duration
            
            formatted = {
                "text": transcription,
                "result": [
                    {
                        "word": word,
                        "start": round(i * step, 2),
                        "end": round((i + 1) * step, 2),
                        "confidence": 1.0
                    }
                    for i, word in enumerate(words)
                ]
            }
            
            logging.info(f"Распознано {len(formatted['result'])} слов")
            logging.info(f"Текст: {transcription[:200]}...")
            
            return formatted
            
        except Exception as e:
            logging.error(f"Ошибка транскрипции: {e}", exc_info=True)
            return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bond005/wav2vec2-large-ru-golos",
                       help="Модель Wav2Vec2")
    parser.add_argument("--use-gpu", action="store_true", 
                       help="Использовать GPU если доступен")
    parser.add_argument("--host", default="localhost", help="Хост")
    parser.add_argument("--port", type=int, default=2702, help="Порт")
    args = parser.parse_args()
    
    server = Wav2Vec2Server(model_name=args.model, use_gpu=args.use_gpu)
    
    async with websockets.serve(
        server.handle_connection, 
        args.host, 
        args.port,
        ping_interval=None,
        close_timeout=10
    ):
        logging.info(f"Wav2Vec2 сервер на ws://{args.host}:{args.port}")
        logging.info(f"Модель: {args.model}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())