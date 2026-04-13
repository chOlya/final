import { useState, useRef, useEffect } from 'react';
import diff_match_patch from 'diff-match-patch';

type FinalKind = 'pending' | 'correct' | 'error' | 'missed';
type ModelType = 'vosk' | 'whisper' | 'wav2vec2';

export default function AudioFileRecognizer() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [partial, setPartial] = useState('');
  const [wordColors, setWordColors] = useState<string[]>([]);
  const [wordTimestamps, setWordTimestamps] = useState<any[]>([]);
  const [currentPlayingWordIndex, setCurrentPlayingWordIndex] = useState(-1);
  const [recognitionStatus, setRecognitionStatus] = useState('');
  const [debugLog, setDebugLog] = useState<string[]>([]);
  const [audioDuration, setAudioDuration] = useState(0);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [isAudioReady, setIsAudioReady] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentSegment, setCurrentSegment] = useState(0);
  const [totalSegments, setTotalSegments] = useState(0);
  const [wordAnalysis, setWordAnalysis] = useState<any[]>([]);

  const [objectiveStats, setObjectiveStats] = useState({
    correct: 0,
    error: 0,
    missed: 0,
    extra: 0,
    totalReference: 0,
    totalRecognized: 0,
    accuracy: 0
  });

  const [sentences, setSentences] = useState<any[]>([]);
  const [referenceText, setReferenceText] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const isHighlightingRef = useRef(false);
  const wordMappingRef = useRef<any[]>([]);           // финальное выравнивание слов
  const audioUrlRef = useRef<string | null>(null);
  const isCancelledRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const allResultsRef = useRef<any[]>([]);            // сырые/объединённые результаты Vosk
  const audioBufferRef = useRef<AudioBuffer | null>(null);
  const dmpRef = useRef<any>(null);                   // экземпляр diff_match_patch
  const wordColorsRef = useRef<string[]>([]);
  // После финализации цвет не меняется (кроме того что CURRENT только для «живого» слова)
  const finalStateRef = useRef<FinalKind[]>([]);

  const referenceWordsRef = useRef<string[]>([]);     // все слова эталона по порядку
  const [selectedModel, setSelectedModel] = useState<ModelType>('vosk');

  const CHUNK_DURATION = 10;       // длина чанка аудио (сек)
  const MAX_RECONNECT_ATTEMPTS = 5;

  const COLORS = {
    CURRENT: '#2196f3',
    CORRECT: '#4caf50',
    ERROR: '#ff9800',
    MISSED: '#f44336',
    PENDING: '#9e9e9e'
  };

  const THRESHOLDS = {
    CORRECT: 80,
    ERROR: 40
  };

  const TIME_WINDOW_LOCAL = 2.0;   // локальное окно по времени для улучшения match
  const TIME_WINDOW_GLOBAL = 6.0;  // максимальный допустимый сдвиг по времени для глобального match

  const WORD_RE = /\p{L}+/u;       // “слово” = последовательность букв (любой алфавит)
  const SIM_THRESHOLD_LOCAL = 60;
  const SIM_THRESHOLD_GLOBAL = 70;

  const UPDATE_INTERVAL_MS = 50;

  // ---------- Шумоподавление ----------
  const applyNoiseReduction = async (audioBuffer: AudioBuffer): Promise<AudioBuffer> => {
    addDebugLog('Применение шумоподавления...');
    const offlineContext = new OfflineAudioContext(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      audioBuffer.sampleRate
    );
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    const highPass = offlineContext.createBiquadFilter();
    highPass.type = 'highpass';
    highPass.frequency.value = 80;
    highPass.Q.value = 0.7;
    const lowPass = offlineContext.createBiquadFilter();
    lowPass.type = 'lowpass';
    lowPass.frequency.value = 8000;
    lowPass.Q.value = 0.7;
    const compressor = offlineContext.createDynamicsCompressor();
    compressor.threshold.value = -24;
    compressor.knee.value = 12;
    compressor.ratio.value = 12;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.25;
    const gain = offlineContext.createGain();
    gain.gain.value = 1.2;
    source.connect(highPass);
    highPass.connect(lowPass);
    lowPass.connect(compressor);
    compressor.connect(gain);
    gain.connect(offlineContext.destination);
    source.start();
    const renderedBuffer = await offlineContext.startRendering();
    addDebugLog('Шумоподавление применено');
    return renderedBuffer;
  };

  const enhanceAudioForVosk = async (audioBuffer: AudioBuffer): Promise<AudioBuffer> => {
    addDebugLog('Улучшение качества аудио для распознавания...');
    return applyNoiseReduction(audioBuffer);
  };

  // Обновление эталонного текста
  const updateReferenceText = (newText: string) => {
    setReferenceText(newText);

    // Бьём эталонный текст на предложения и слова
    const sentencesArray = segmentTextBySentences(newText);
    setSentences(sentencesArray);

    const allWords: string[] = [];
    sentencesArray.forEach((sentence) => {
      sentence.words.forEach((word: string) => allWords.push(word));
    });

    referenceWordsRef.current = allWords;

    // Изначально все слова — в состоянии “ожидание”
    const initialColors = Array(allWords.length).fill(COLORS.PENDING);
    setWordColors(initialColors);
    wordColorsRef.current = initialColors;
    finalStateRef.current = Array(allWords.length).fill('pending');
    addDebugLog(`Текст разбит на ${sentencesArray.length} предложений (${allWords.length} слов)`);
  };

  useEffect(() => {
    // Настраиваем diff-match-patch
    dmpRef.current = new diff_match_patch();
    dmpRef.current.Diff_Timeout = 1.5;
    dmpRef.current.Match_Threshold = 0.3;
    dmpRef.current.Match_Distance = 500;
    dmpRef.current.Patch_DeleteThreshold = 0.4;

    // Бьём эталонный текст на предложения и слова
    updateReferenceText(referenceText);
  }, []);

  // ---------- Очистка при размонтировании ----------
  useEffect(() => {
    return () => {
      if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      if (wsRef.current) wsRef.current.close();
      if (audioContextRef.current) audioContextRef.current.close();
      isCancelledRef.current = true;
    };
  }, []);

  // ---------- Утилиты: лог, нормализация слов, diff-помощники ----------

  const addDebugLog = (message: string) => {
    console.log(message);
    setDebugLog((prev) => [
      ...prev.slice(-30),
      `${new Date().toLocaleTimeString()}: ${message}`
    ]);
  };

  // Единая нормализация слов для сравнения
  const normalizeWord = (word: string) => {
    if (!word) return '';
    return word
      .toLowerCase()
      .replace(/[.,!?;:()\[\]{}"'-]/g, '')
      .replace(/ё/g, 'е')
      .replace(/\s+/g, ' ')  // Нормализуем пробелы
      .trim();
  };

  // Похожесть двух слов по символам, на базе diff-match-patch
  const similarityPercent = (dmp: any, a: string, b: string): number => {
    const normA = normalizeWord(a);
    const normB = normalizeWord(b);
    if (!normA && !normB) return 100;
    if (!normA || !normB) return 0;

    const diffs = dmp.diff_main(normA, normB);
    dmp.diff_cleanupSemantic(diffs);

    let equalLength = 0;
    for (const [op, text] of diffs as [number, string][]) {
      if (op === 0) equalLength += text.length;
    }
    const total = normA.length + normB.length;
    return total ? (equalLength * 2 * 100) / total : 0;
  };

  // Преобразование списка токенов в строку уникальных символов (паттерн dmp)
  const tokensToChars = (tokens: string[], token2char: Record<string, string> = {}) => {
    const chars: string[] = [];
    for (const tok of tokens) {
      if (!token2char[tok]) {
        token2char[tok] = String.fromCharCode(Object.keys(token2char).length + 1);
      }
      chars.push(token2char[tok]);
    }
    return { text: chars.join(''), map: token2char };
  };

  // Базовое выравнивание списков слов (эталон/распознанные) через diff
  const alignWithDiff = (refTokensNorm: string[], recTokensNorm: string[], dmp: any) => {
    const { text: refChars, map } = tokensToChars(refTokensNorm);
    const { text: recChars } = tokensToChars(recTokensNorm, map);

    const diffs = dmp.diff_main(refChars, recChars);
    dmp.diff_cleanupSemantic(diffs);

    const refToRecIdx = new Array(refTokensNorm.length).fill(null) as (number | null)[];
    const evalRecIdx = new Array(refTokensNorm.length).fill(0) as number[];

    let refI = 0;
    let recI = 0;

    for (const [op, str] of diffs as [number, string][]) {
      const length = str.length;

      if (op === 0) {
        // совпадающие токены
        for (let k = 0; k < length; k++) {
          refToRecIdx[refI] = recI;
          evalRecIdx[refI] = recI;
          refI++;
          recI++;
        }
      } else if (op === -1) {
        // DELETE – токены только в эталоне
        for (let k = 0; k < length; k++) {
          evalRecIdx[refI] = recI;
          refI++;
        }
      } else if (op === 1) {
        // INSERT – токены только в распознанном тексте
        recI += length;
      }
    }

    for (let i = 0; i < evalRecIdx.length; i++) {
      if (evalRecIdx[i] == null) evalRecIdx[i] = recI;
    }

    return { refToRecIdx, evalRecIdx };
  };

  // Локальное улучшение: для слов без пары ищем кандидата в окне по времени
  const improveWithMatchLocal = (
    refTokens: string[],
    recWords: any[],
    refToRecIdx: (number | null)[],
    evalRecIdx: number[],
    dmp: any
  ) => {
    if (!recWords.length) return { refToRecIdx, evalRecIdx };

    const used = new Set<number>();
    refToRecIdx.forEach((idx) => {
      if (idx != null) used.add(idx);
    });

    const duration = recWords.length ? recWords[recWords.length - 1].end : 0;

    for (let i = 0; i < refTokens.length; i++) {
      if (refToRecIdx[i] != null) continue;

      let approxIdx = evalRecIdx[i];
      if (approxIdx < 0) approxIdx = 0;
      if (approxIdx >= recWords.length) approxIdx = recWords.length - 1;

      const expectedTime =
        recWords[approxIdx]?.start ?? (duration ? (i / refTokens.length) * duration : 0);
      const candidates: { j: number; score: number; sim: number }[] = [];
      for (let j = 0; j < recWords.length; j++) {
        if (used.has(j)) continue;
        const rec = recWords[j];
        if (Math.abs(rec.start - expectedTime) > TIME_WINDOW_LOCAL) continue;
        const sim = similarityPercent(dmp, refTokens[i], rec.word);
        if (sim > 0) {
          const timeBonus = Math.max(0, 30 - Math.abs(rec.start - expectedTime) * 10);
          candidates.push({ j, score: sim + timeBonus, sim });
        }
      }
      if (!candidates.length) continue;
      candidates.sort((a, b) => b.score - a.score);
      const best = candidates[0];
      if (best.sim >= SIM_THRESHOLD_LOCAL) {
        refToRecIdx[i] = best.j;
        used.add(best.j);
      }
    }
    return { refToRecIdx, evalRecIdx };
  };

  // Глобальный fallback: для оставшихся слов ищем кандидата по всей траектории,
  // но ограничиваемся окном по времени, чтобы “Слово1” не приклеилось к позднему “Слово1”.
  const globalFallbackMatch = (
    referenceWords: string[],
    recWords: any[],
    refToRecIdx: (number | null)[],
    dmp: any
  ) => {
    if (!recWords.length) return refToRecIdx;

    const used = new Set<number>();
    refToRecIdx.forEach((idx) => {
      if (idx != null) used.add(idx);
    });
    const duration = recWords.length ? recWords[recWords.length - 1].end || 0 : 0;
    for (let i = 0; i < referenceWords.length; i++) {
      if (refToRecIdx[i] != null) continue;
      const expectedTime = duration ? (i / referenceWords.length) * duration : 0;
      let bestJ = -1;
      let bestScore = 0;
      let bestSim = 0;

      for (let j = 0; j < recWords.length; j++) {
        if (used.has(j)) continue;
        const rec = recWords[j];

        const timeDiff = Math.abs(rec.start - expectedTime);
        if (timeDiff > TIME_WINDOW_GLOBAL) continue;

        const sim = similarityPercent(dmp, referenceWords[i], rec.word);
        if (!sim) continue;
        const score = sim - timeDiff * 10;
        if (score > bestScore) {
          bestScore = score;
          bestSim = sim;
          bestJ = j;
        }
      }

      if (bestJ !== -1 && bestSim >= SIM_THRESHOLD_GLOBAL) {
        refToRecIdx[i] = bestJ;
        used.add(bestJ);
      }
    }

    return refToRecIdx;
  };

  const enrichMappingWithTimeWindows = (mapping: any[], totalDuration: number) => {
    const n = mapping.length;
    for (let i = 0; i < n; i++) {
      const m = mapping[i];
      const vosk = m.voskWord;
      if (vosk) {
        m.expectedStart = vosk.start;
        m.expectedEnd = vosk.end;
        continue;
      }
      const t0 = m.expectedTime ?? 0;
      let t1 = totalDuration;
      if (i + 1 < n) {
        const next = mapping[i + 1];
        if (next.voskWord) t1 = Math.min(t1, next.voskWord.start);
        else t1 = Math.min(t1, next.expectedTime ?? totalDuration);
      }
      const span = Math.max(0.15, t1 - t0);
      let gapStart = i;
      while (gapStart > 0 && !mapping[gapStart - 1].voskWord) gapStart--;
      let gapEnd = i;
      while (gapEnd < n - 1 && !mapping[gapEnd + 1].voskWord) gapEnd++;
      const cnt = gapEnd - gapStart + 1;
      const segLen = span / Math.max(1, cnt);
      for (let k = gapStart; k <= gapEnd; k++) {
        const off = k - gapStart;
        const es = t0 + off * segLen;
        const ee = Math.min(t1, es + segLen * 0.95);
        mapping[k].expectedStart = es;
        mapping[k].expectedEnd = ee;
      }
    }
    for (let i = 0; i < n; i++) {
      const m = mapping[i];
      if (m.voskWord) continue;
      if (m.expectedStart == null) m.expectedStart = m.expectedTime ?? 0;
      if (m.expectedEnd == null) m.expectedEnd = Math.min(totalDuration, (m.expectedStart as number) + 0.4);
    }
    return mapping;
  };


  // Главное выравнивание: diff → локальный match → глобальный match → монотонное время

  const buildDiffMatchAlignment = (
    referenceWords: string[],
    recWords: any[],
    dmp: any,
    setObjectiveStats: (x: any) => void,
    setWordAnalysis: (x: any[]) => void,
    totalDuration: number
  ) => {
    const refNorm = referenceWords.map(normalizeWord);
    const recNorm = recWords.map((w) => normalizeWord(w.word));

    const { refToRecIdx, evalRecIdx } = alignWithDiff(refNorm, recNorm, dmp);
    const improvedLocal = improveWithMatchLocal(
      referenceWords,
      recWords,
      refToRecIdx,
      evalRecIdx,
      dmp
    );
    const finalRefToRec = globalFallbackMatch(
      referenceWords,
      recWords,
      improvedLocal.refToRecIdx,
      dmp
    );

    const duration = recWords.length ? recWords[recWords.length - 1].end : 0;

    const mapping: any[] = [];
    const analysis: any[] = [];
    let correct = 0,
      error = 0,
      missed = 0;

    let lastRecIdx = -1;  // предотвращаем “откат” по индексам Vosk
    let lastTime = 0;     // предотвращаем “откат” по ожидаемому времени

    for (let i = 0; i < referenceWords.length; i++) {
      const refWord = referenceWords[i];
      let matchIdx = finalRefToRec[i];

      // если матч указывает на слово раньше уже использованного — отбрасываем
      if (matchIdx != null && matchIdx < lastRecIdx) {
        matchIdx = null;
      }

      if (matchIdx == null) {
        // слова без пары — считаем пропущенными, но всё равно задаём
        // ожидаемое время (монотонное по i)
        const expectedBase = duration
          ? (i / referenceWords.length) * duration
          : 0;
        const expectedTime = Math.max(lastTime, expectedBase);

        mapping.push({
          referenceIndex: i,
          referenceWord: refWord,
          voskWord: null,
          expectedTime,
          actualTime: null,
          timeDiff: null,
          alignment: 'diff',
          status: 'missed'
        });

        analysis.push({
          position: i + 1,
          expectedTime: expectedTime.toFixed(1),
          expected: refWord,
          actual: '(не распознано)',
          similarity: 0,
          status: 'missed',
          color: COLORS.MISSED,
          changes: []
        });

        lastTime = expectedTime;
        missed++;
      } else {
        // есть валидное сопоставление с Vosk-словом
        const voskWord = recWords[matchIdx];
        const baseTime = voskWord.start;
        const expectedTime = Math.max(lastTime, baseTime);
        const actualTime = baseTime;
        const timeDiff = expectedTime - actualTime;

        const sim = similarityPercent(dmp, refWord, voskWord.word);
        let status: 'correct' | 'error' | 'missed';
        let color: string;

        if (sim >= THRESHOLDS.CORRECT) {
          status = 'correct';
          color = COLORS.CORRECT;
          correct++;
        } else if (sim >= THRESHOLDS.ERROR) {
          status = 'error';
          color = COLORS.ERROR;
          error++;
        } else {
          status = 'missed';
          color = COLORS.MISSED;
          missed++;
        }

        mapping.push({
          referenceIndex: i,
          referenceWord: refWord,
          voskWord,
          expectedTime,
          actualTime,
          timeDiff,
          alignment: 'diff+match',
          status,
          similarity: sim
        });

        analysis.push({
          position: i + 1,
          expectedTime: expectedTime.toFixed(1),
          actualTime: actualTime.toFixed(1),
          timeDiff: timeDiff.toFixed(1),
          expected: refWord,
          actual: voskWord.word,
          similarity: sim,
          status,
          color,
          changes: []
        });

        lastRecIdx = matchIdx;
        lastTime = expectedTime;
      }
    }

    enrichMappingWithTimeWindows(mapping, totalDuration || duration);

    const stats = {
      correct,
      error,
      missed,
      extra: 0,
      totalReference: referenceWords.length,
      totalRecognized: recWords.length,
      accuracy: referenceWords.length ? (correct / referenceWords.length) * 100 : 0
    };

    setObjectiveStats(stats);
    setWordAnalysis(analysis);

    return mapping;
  };

  // Разбиение текста на предложения и слова
  const segmentTextBySentences = (text: string) => {
    const sentenceRegex = /[^.!?]+[.!?]+/g;
    const matches = text.match(sentenceRegex) || [];

    if (matches.length === 0) {
      return chunkArray(text.split(' '), 15).map((chunk, index) => ({
        id: index,
        text: chunk.join(' '),
        words: chunk,
        expectedDuration: chunk.length * 0.4
      }));
    }

    return matches.map((sentence, index) => {
      const words = sentence.split(' ').filter((w) => w.length > 0);
      return {
        id: index,
        text: sentence.trim(),
        words: words,
        expectedDuration: words.length * 0.4 + 0.5
      };
    });
  };

  const chunkArray = (array: any[], size: number) => {
    const chunks: any[] = [];
    for (let i = 0; i < array.length; i += size) chunks.push(array.slice(i, i + size));
    return chunks;
  };

  // Разбиение AudioBuffer на перекрывающиеся чанки
  const splitAudioIntoChunks = (
    audioBuffer: AudioBuffer,
    chunkDuration: number,
    overlapDuration = 1.0
  ) => {
    const chunks: any[] = [];
    const sampleRate = audioBuffer.sampleRate;
    const totalSamples = audioBuffer.length;
    const chunkSamples = chunkDuration * sampleRate;
    const overlapSamples = overlapDuration * sampleRate;
    const stepSamples = chunkSamples - overlapSamples;

    let startSample = 0;
    let chunkIndex = 0;

    while (startSample < totalSamples) {
      const endSample = Math.min(startSample + chunkSamples, totalSamples);
      const actualChunkSamples = endSample - startSample;

      // слишком маленький остаток — домержим к последнему чанку
      if (actualChunkSamples < sampleRate * 2) {
        if (chunks.length > 0) {
          const lastChunk = chunks[chunks.length - 1];
          lastChunk.endSample = totalSamples;
          lastChunk.duration = (totalSamples - lastChunk.startSample) / sampleRate;
        }
        break;
      }

      chunks.push({
        index: chunkIndex,
        startSample,
        endSample,
        startTime: startSample / sampleRate,
        endTime: endSample / sampleRate,
        duration: actualChunkSamples / sampleRate,
        buffer: extractAudioSegment(audioBuffer, startSample, endSample)
      });

      startSample += stepSamples;
      chunkIndex++;
    }

    addDebugLog(`Аудио разбито на ${chunks.length} чанков по ${chunkDuration}с`);
    return chunks;
  };
  // Вырезаем сегмент AudioBuffer
  const extractAudioSegment = (audioBuffer: AudioBuffer, startSample: number, endSample: number) => {
    const length = endSample - startSample;
    if (length <= 0) return null;

    const segmentBuffer = audioContextRef.current!.createBuffer(
      audioBuffer.numberOfChannels,
      length,
      audioBuffer.sampleRate
    );

    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel);
      const segmentChannelData = segmentBuffer.getChannelData(channel);

      for (let i = 0; i < length; i++) {
        if (startSample + i < channelData.length) segmentChannelData[i] = channelData[startSample + i];
      }
    }
    return segmentBuffer;
  };

  // Используем динамический URL для WebSocket
  const processChunkWithRetry = async (
    chunk: any,
    chunkIndex: number,
    totalChunks: number,
    retryCount = 0
  ) => {
    return new Promise<any[]>((resolve) => {
      addDebugLog(`Чанк ${chunkIndex + 1}/${totalChunks} (${selectedModel.toUpperCase()}) (попытка ${retryCount + 1})`);

      // Используем динамический URL в зависимости от выбранной модели
      const ws = new WebSocket(getWebSocketUrl());
      let chunkResults: any[] = [];
      let complete = false;

      const timeoutId = setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) ws.close();
        if (!complete && retryCount < MAX_RECONNECT_ATTEMPTS) {
          setTimeout(async () => {
            resolve(await processChunkWithRetry(chunk, chunkIndex, totalChunks, retryCount + 1));
          }, 1000);
        } else resolve(chunkResults);
      }, chunk.duration * 1000 + 10000);

      ws.onopen = () => {
        // Для Vosk нужна конфигурация, для Whisper нет
        if (selectedModel === 'vosk') {
          ws.send(JSON.stringify({ config: { sample_rate: 16000, words: true, max_alternatives: 0 } }));
        }

        setTimeout(() => {
          const channelData = chunk.buffer.getChannelData(0);
          const pcm16 = new Int16Array(channelData.length);
          for (let i = 0; i < channelData.length; i++) {
            pcm16[i] = Math.max(-32768, Math.min(32767, channelData[i] * 32767));
          }
          ws.send(new Uint8Array(pcm16.buffer));
          setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) ws.send('{"eof" : 1}');
          }, 500);
        }, 200);
      };

      ws.onmessage = (e) => {
        try {
          let data: any;
          if (typeof e.data === 'string') data = JSON.parse(e.data);
          else data = JSON.parse(new TextDecoder().decode(e.data as ArrayBuffer));
          // финальные результаты чанка с разметкой слов
          if (data.result) {
            data.result.forEach((word: any) => {
              chunkResults.push({
                word: word.word,
                start: (word.start || 0) + chunk.startTime,
                end: (word.end || 0) + chunk.startTime,
                confidence: word.confidence || 1.0,
                chunkIndex
              });
            });
          }

          // по наличию text считаем чанк завершённым
          if (data.text && !complete) {
            complete = true;
            clearTimeout(timeoutId);
            ws.close();
            setProgress(Math.floor(((chunkIndex + 1) / totalChunks) * 100));
            setCurrentSegment(chunkIndex + 1);
            resolve(chunkResults);
          }
        } catch { }
      };
      ws.onerror = () => {
        if (!complete && retryCount < MAX_RECONNECT_ATTEMPTS) {
          clearTimeout(timeoutId);
          ws.close();
          setTimeout(async () => {
            resolve(await processChunkWithRetry(chunk, chunkIndex, totalChunks, retryCount + 1));
          }, 1000);
        } else {
          clearTimeout(timeoutId);
          resolve(chunkResults);
        }
      };
      ws.onclose = () => {
        if (!complete) {
          clearTimeout(timeoutId);
          resolve(chunkResults);
        }
      };
    });
  };

  // Последовательная обработка всех чанков
  const processAllChunks = async (chunks: any[]) => {
    const allResults: any[] = [];
    const totalChunks = chunks.length;

    setTotalSegments(totalChunks);

    for (let i = 0; i < chunks.length; i++) {
      if (isCancelledRef.current) break;
      const chunkResults = await processChunkWithRetry(chunks[i], i, totalChunks);
      allResults.push(...chunkResults);

      allResultsRef.current = allResults;
      await new Promise((r) => setTimeout(r, 500));
    }

    return allResults;
  };

  // Удаление дублей слов (одинаковое слово, почти то же время)
  const deduplicateResults = (results: any[], overlapThreshold = 0.5) => {
    if (results.length === 0) return results;

    const sorted = [...results].sort((a, b) => a.start - b.start);
    const unique: any[] = [];
    const used = new Set<number>();

    for (let i = 0; i < sorted.length; i++) {
      const current = sorted[i];
      let isDuplicate = false;

      for (let j = 0; j < unique.length; j++) {
        const existing = unique[j];

        if (
          normalizeWord(current.word) === normalizeWord(existing.word) &&
          Math.abs(current.start - existing.start) < overlapThreshold
        ) {
          isDuplicate = true;
          break;
        }
      }

      if (!isDuplicate && !used.has(i)) {
        unique.push(current);
        used.add(i);
      }
    }
    addDebugLog(`Удаление дубликатов: было ${results.length}, стало ${unique.length}`);
    return unique;
  };

  /** Финальный цвет по статусу (один раз) */
  const statusToFinalColor = (status: string): string => {
    if (status === 'correct') return COLORS.CORRECT;
    if (status === 'error') return COLORS.ERROR;
    return COLORS.MISSED;
  };

  // Подсветка слов в зависимости от текущего времени аудио
  const updateWordHighlighting = (currentTime: number) => {
    if (!wordMappingRef.current.length || !audioDuration) return;

    const mapping = wordMappingRef.current;
    const n = referenceWordsRef.current.length;
    const EPS = 0.02;

    while (finalStateRef.current.length < n) finalStateRef.current.push('pending');
    finalStateRef.current.length = n;

    const getEntry = (i: number) => mapping.find((x: any) => x.referenceIndex === i);

    const getSpan = (i: number) => {
      const m = getEntry(i);
      if (!m) {
        return { start: 0, end: 0.35 };
      }
      if (m.voskWord) {
        return { start: m.voskWord.start, end: m.voskWord.end };
      }
      const start = m.expectedStart ?? m.expectedTime ?? 0;
      const end = m.expectedEnd ?? Math.min(audioDuration, start + 0.35);
      return { start, end };
    };

    const finalizeOne = (i: number) => {
      if (finalStateRef.current[i] !== 'pending') return;
      const m = getEntry(i);
      if (!m || !m.voskWord) {
        finalStateRef.current[i] = 'missed';
        return;
      }
      if (m.status === 'correct') finalStateRef.current[i] = 'correct';
      else if (m.status === 'error') finalStateRef.current[i] = 'error';
      else finalStateRef.current[i] = 'missed';
    };

    const finalizedColor = (i: number) => {
      const f = finalStateRef.current[i];
      if (f === 'correct') return COLORS.CORRECT;
      if (f === 'error') return COLORS.ERROR;
      return COLORS.MISSED;
    };

    // 1) Прошёл конец слота — фиксируем по порядку (пропуск без vosk → красный)
    for (let i = 0; i < n; i++) {
      if (finalStateRef.current[i] !== 'pending') continue;
      const { end } = getSpan(i);
      if (currentTime > end + EPS) finalizeOne(i);
    }

    // 2) Первая ещё незафиксированная позиция — единственная, где допускается CURRENT
    let frontier = -1;
    for (let i = 0; i < n; i++) {
      if (finalStateRef.current[i] === 'pending') {
        frontier = i;
        break;
      }
    }

    const newColors = new Array<string>(n);
    let currentIdx = -1;

    for (let i = 0; i < n; i++) {
      const fin = finalStateRef.current[i];
      if (fin !== 'pending') {
        newColors[i] = finalizedColor(i);
        continue;
      }

      if (frontier === -1) {
        newColors[i] = COLORS.PENDING;
        continue;
      }

      if (i < frontier) {
        // не должно случаться, если финализация выше согласована
        newColors[i] = COLORS.PENDING;
        continue;
      }

      if (i > frontier) {
        newColors[i] = COLORS.PENDING;
        continue;
      }

      // i === frontier
      const { start, end } = getSpan(i);
      if (currentTime < start - EPS) {
        newColors[i] = COLORS.PENDING;
      } else if (currentTime <= end + EPS) {
        newColors[i] = COLORS.CURRENT;
        currentIdx = i;
      } else {
        finalizeOne(i);
        newColors[i] = finalizedColor(i);
      }
    }

    let changed = false;
    for (let i = 0; i < n; i++) {
      if (newColors[i] !== wordColorsRef.current[i]) {
        changed = true;
        break;
      }
    }
    if (changed) {
      setWordColors(newColors);
      wordColorsRef.current = newColors;
    }
    setCurrentPlayingWordIndex(currentIdx);
    setAudioCurrentTime(currentTime);
  };

  // Цикл подсветки на базе requestAnimationFrame
  const startHighlightingLoop = () => {
    if (isHighlightingRef.current) return;
    if (!audioRef.current || audioRef.current.paused) return;

    isHighlightingRef.current = true;
    let lastUpdate = 0;
    const loop = () => {
      if (!audioRef.current || !isHighlightingRef.current) {
        if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
        return;
      }
      const audio = audioRef.current;
      if (audio.paused) {
        isHighlightingRef.current = false;
        if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
        return;
      }
      const now = Date.now();
      if (now - lastUpdate >= UPDATE_INTERVAL_MS) {
        updateWordHighlighting(audio.currentTime);
        lastUpdate = now;
      }
      animationFrameRef.current = requestAnimationFrame(loop);
    };

    animationFrameRef.current = requestAnimationFrame(loop);
  };

  const handleAudioPlay = () => {
    if (audioRef.current && audioRef.current.currentTime < 0.5) {
      const resetColors = Array(referenceWordsRef.current.length).fill(COLORS.PENDING);
      setWordColors(resetColors);
      wordColorsRef.current = resetColors;
      finalStateRef.current = Array(referenceWordsRef.current.length).fill('pending');
      setCurrentPlayingWordIndex(-1);
    }
    startHighlightingLoop();
  };

  const handleAudioPause = () => {
    isHighlightingRef.current = false;
  };

  const handleAudioEnded = () => {
    // в конце просто показываем финальный статус всех слов
    isHighlightingRef.current = false;
    const n = referenceWordsRef.current.length;
    const newColors = [...wordColorsRef.current];
    for (let i = 0; i < n; i++) {
      const item = wordMappingRef.current.find((m) => m.referenceIndex === i);
      if (!item) continue;
      finalStateRef.current[i] =
        item.status === 'correct' ? 'correct' : item.status === 'error' ? 'error' : 'missed';
      newColors[i] = statusToFinalColor(item.status);
    }
    setWordColors(newColors);
    wordColorsRef.current = newColors;
  };

  const handleAudioLoadedMetadata = () => {
    const audio = audioRef.current;
    if (!audio) return;
    setAudioDuration(audio.duration || 0);
    setIsAudioReady(true);
  };

  const readFileAsArrayBuffer = (file: File) => {
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.onerror = () => reject(new Error('Ошибка чтения файла'));
      reader.readAsArrayBuffer(file);
    });
  };

  // Главная функция: загрузка файла, декодирование, разбивка, распознавание, выравнивание
  const processFile = async (selectedFile: File) => {
    console.clear();
    addDebugLog('Начало обработки файла (diff → match + время)');

    isCancelledRef.current = false;
    allResultsRef.current = [];

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    if (wsRef.current) wsRef.current.close();
    if (audioContextRef.current) audioContextRef.current.close();

    setFile(selectedFile);
    setIsProcessing(true);
    setRecognitionStatus('Подготовка файла...');
    setTranscription('');
    setPartial('');
    setWordColors(Array(referenceWordsRef.current.length).fill(COLORS.PENDING));
    setWordTimestamps([]);
    setCurrentPlayingWordIndex(-1);
    setAudioDuration(0);
    setAudioCurrentTime(0);
    setIsAudioReady(false);
    setProgress(0);
    setCurrentSegment(0);
    setTotalSegments(0);
    setDebugLog([]);
    setWordAnalysis([]);
    setObjectiveStats({
      correct: 0,
      error: 0,
      missed: 0,
      extra: 0,
      totalReference: 0,
      totalRecognized: 0,
      accuracy: 0
    });

    wordMappingRef.current = [];
    isHighlightingRef.current = false;
    wordColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS.PENDING);
    finalStateRef.current = Array(referenceWordsRef.current.length).fill('pending');

    audioUrlRef.current = URL.createObjectURL(selectedFile);

    try {
      // 1. Декодируем аудио в AudioBuffer
      setRecognitionStatus('Чтение и декодирование аудио...');
      const arrayBuffer = await readFileAsArrayBuffer(selectedFile);

      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000
      });
      let audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);

      addDebugLog('🎛️ Обработка шумов радиопереговоров...');
      audioBuffer = await enhanceAudioForVosk(audioBuffer);
      audioBufferRef.current = audioBuffer;

      const duration = audioBuffer.duration;
      addDebugLog(`Аудио: ${duration.toFixed(2)} секунд`);
      setAudioDuration(duration);

      // 2. Разбиваем на чанки
      setRecognitionStatus('Разбиение аудио на чанки...');
      const chunks = splitAudioIntoChunks(audioBuffer, CHUNK_DURATION, 1.0);

      // 3. Распознаём каждый чанк через Vosk
      setRecognitionStatus('Распознавание речи...');
      const rawResults = await processAllChunks(chunks);

      if (isCancelledRef.current) {
        addDebugLog('Обработка отменена');
        setIsProcessing(false);
        return;
      }

      // 4. Удаляем дубликаты
      const uniqueResults = deduplicateResults(rawResults);

      // 5. Выравниваем эталон и Vosk-слова
      setRecognitionStatus('Выравнивание (diff → match)...');
      const finalMapping = buildDiffMatchAlignment(
        referenceWordsRef.current,
        uniqueResults,
        dmpRef.current,
        setObjectiveStats,
        setWordAnalysis,
        duration
      );

      wordMappingRef.current = finalMapping;

      // 6. Сохраняем распознанный текст и таймстемпы для отображения
      setWordTimestamps(uniqueResults);
      setTranscription(uniqueResults.map((w) => w.word).join(' '));

      setProgress(100);
      setIsProcessing(false);
      setRecognitionStatus(`Распознавание завершено! ${uniqueResults.length} слов через ${selectedModel.toUpperCase()}`);
    } catch (error: any) {
      addDebugLog(`Критическая ошибка: ${error.message}`);
      console.error(error);
      setRecognitionStatus(`Ошибка: ${error.message}`);
      setIsProcessing(false);
    }
  };

  const handleNewAttempt = () => {
    isCancelledRef.current = true;
    isHighlightingRef.current = false;
    if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
    if (wsRef.current) wsRef.current.close();
    if (audioContextRef.current) audioContextRef.current.close();
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }

    // Полный сброс состояния
    setFile(null);
    setIsProcessing(false);
    setRecognitionStatus('');
    setTranscription('');
    setPartial('');
    setWordColors(Array(referenceWordsRef.current.length).fill(COLORS.PENDING));
    setWordTimestamps([]);
    setCurrentPlayingWordIndex(-1);
    setAudioDuration(0);
    setAudioCurrentTime(0);
    setIsAudioReady(false);
    setProgress(0);
    setCurrentSegment(0);
    setTotalSegments(0);
    setDebugLog([]);
    setWordAnalysis([]);
    setObjectiveStats({
      correct: 0,
      error: 0,
      missed: 0,
      extra: 0,
      totalReference: 0,
      totalRecognized: 0,
      accuracy: 0
    });

    wordMappingRef.current = [];
    allResultsRef.current = [];
    audioBufferRef.current = null;
    wordColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS.PENDING);
    finalStateRef.current = Array(referenceWordsRef.current.length).fill('pending');

    addDebugLog('Новая попытка');
  };

  const getAudioUrl = () => audioUrlRef.current || '';

  const getWebSocketUrl = () => {
    if (selectedModel === 'whisper') {
      return 'ws://localhost:2701';  // Whisper на 2701
    } else if (selectedModel === 'wav2vec2') {
      return 'ws://localhost:2702';  // Wav2Vec2 на 2702
    } else {
      return 'ws://localhost:2700';  // Vosk на 2700
    }
  };

  // Компонент переключателя моделей
  const ModelSelector = () => (
    <div style={{
      display: 'flex',
      gap: '1rem',
      marginBottom: '1rem',
      padding: '0.75rem',
      background: 'white',
      borderRadius: '12px',
      border: '1px solid #e2e8f0',
      justifyContent: 'center',
      flexWrap: 'wrap'
    }}>
      <label style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        cursor: isProcessing ? 'not-allowed' : 'pointer',
        opacity: isProcessing ? 0.6 : 1,
        padding: '0.5rem 1rem',
        borderRadius: '8px',
        background: selectedModel === 'vosk' ? '#fef3c7' : 'transparent'
      }}>
        <input
          type="radio"
          name="model"
          value="vosk"
          checked={selectedModel === 'vosk'}
          onChange={(e) => setSelectedModel(e.target.value as ModelType)}
          disabled={isProcessing}
        />
        <span>Vosk</span>
      </label>

      <label style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        cursor: isProcessing ? 'not-allowed' : 'pointer',
        opacity: isProcessing ? 0.6 : 1,
        padding: '0.5rem 1rem',
        borderRadius: '8px',
        background: selectedModel === 'whisper' ? '#e0f2fe' : 'transparent'
      }}>
        <input
          type="radio"
          name="model"
          value="whisper"
          checked={selectedModel === 'whisper'}
          onChange={(e) => setSelectedModel(e.target.value as ModelType)}
          disabled={isProcessing}
        />
        <span>Whisper</span>
      </label>

      <label style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        cursor: isProcessing ? 'not-allowed' : 'pointer',
        opacity: isProcessing ? 0.6 : 1,
        padding: '0.5rem 1rem',
        borderRadius: '8px',
        background: selectedModel === 'wav2vec2' ? '#d1fae5' : 'transparent'
      }}>
        <input
          type="radio"
          name="model"
          value="wav2vec2"
          checked={selectedModel === 'wav2vec2'}
          onChange={(e) => setSelectedModel(e.target.value as ModelType)}
          disabled={isProcessing}
        />
        <span>Wav2Vec2</span>
      </label>
    </div>
  );

  // ---------- Основной JSX ----------
  return (
    <div style={{ minHeight: '100vh', background: '#f1f5f9', padding: 0, margin: 0 }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1rem' }}>

        {/* Header */}
        <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: '600', color: '#0f172a', letterSpacing: '-0.02em', marginBottom: '0.25rem' }}>
              Речевой тренажер
            </h1>
            <div style={{ width: '50px', height: '3px', background: '#3b82f6' }} />
          </div>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            {[
              { color: '#3b82f6', label: 'Текущее' },
              { color: '#10b981', label: 'Правильно' },
              { color: '#f59e0b', label: 'Ошибка' },
              { color: '#ef4444', label: 'Пропуск' },
              { color: '#cbd5e1', label: 'Ожидание' }
            ].map((item, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: item.color }} />
                <span style={{ fontSize: '0.75rem', fontWeight: '500', color: '#475569' }}>{item.label}</span>
              </div>
            ))}
          </div>
        </div>

        <ModelSelector />

        {/* {isProcessing && (
          <div style={{
            marginBottom: '1rem',
            padding: '0.5rem',
            background: selectedModel === 'whisper' ? '#e0f2fe' : '#fef3c7',
            borderRadius: '8px',
            fontSize: '0.8rem',
            textAlign: 'center',
            border: `1px solid ${selectedModel === 'whisper' ? '#7dd3fc' : '#fde68a'}`
          }}>
            🔄 Используется модель: <strong>{selectedModel === 'whisper' ? 'Whisper' : 'Vosk'}</strong>
            {selectedModel === 'whisper' ? ' (более точное распознавание)' : ' (быстрое распознавание)'}
          </div>
        )} */}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '1.25rem', marginBottom: '1.25rem', alignItems: 'start' }}>
          <div style={{ minWidth: 0, overflow: 'hidden' }}>
            {!file ? (
              <div style={{
                background: 'white',
                borderRadius: '12px',
                padding: '1.25rem',
                border: '1px solid #e2e8f0'
              }}>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: '600', color: '#475569', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Эталонный текст
                </label>
                <textarea
                  value={referenceText}
                  onChange={(e) => updateReferenceText(e.target.value)}
                  disabled={isProcessing}
                  style={{
                    width: '100%',
                    padding: '0.875rem',
                    border: '1.5px solid #e2e8f0',
                    borderRadius: '10px',
                    fontSize: '1rem',
                    lineHeight: '1.5',
                    color: '#0f172a',
                    background: isProcessing ? '#f8fafc' : 'white',
                    resize: 'vertical',
                    minHeight: '180px',
                    outline: 'none',
                    fontFamily: 'inherit',
                    boxSizing: 'border-box'
                  }}
                  placeholder="Введите текст для анализа..."
                />
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.75rem' }}>
                  {referenceWordsRef.current.length} слов
                </div>
              </div>
            ) : (
              <div style={{
                background: 'white',
                borderRadius: '12px',
                padding: '1.25rem',
                border: '1px solid #e2e8f0'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: '600', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Эталонный текст
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8', background: '#f1f5f9', padding: '0.25rem 0.5rem', borderRadius: '4px' }}>
                    {referenceWordsRef.current.length} слов
                  </div>
                </div>
                <div style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.5rem',
                  maxHeight: '400px',
                  overflowY: 'auto',
                  overflowX: 'hidden'
                }}>
                  {referenceWordsRef.current.map((word, index) => {
                    const color = wordColors[index] || COLORS.PENDING;
                    const styles: Record<string, React.CSSProperties> = {
                      [COLORS.CURRENT]: { background: '#dbeafe', color: '#1e40af', borderColor: '#bfdbfe', fontWeight: 600 },
                      [COLORS.CORRECT]: { background: '#d1fae5', color: '#065f46', borderColor: '#a7f3d0' },
                      [COLORS.ERROR]: { background: '#fed7aa', color: '#9a3412', borderColor: '#fdba74' },
                      [COLORS.MISSED]: { background: '#fee2e2', color: '#991b1b', borderColor: '#fecaca' },
                      [COLORS.PENDING]: { background: '#f8fafc', color: '#cbd5e1', borderColor: '#e2e8f0' }
                    };
                    return (
                      <span
                        key={index}
                        style={{
                          padding: '0.35rem 0.75rem',
                          borderRadius: '8px',
                          fontSize: '0.9rem',
                          fontWeight: color === COLORS.CURRENT ? '600' : '400',
                          border: '1px solid',
                          transition: 'all 0.15s ease',
                          whiteSpace: 'nowrap',
                          ...styles[color],
                          ...(color === COLORS.CURRENT ? { boxShadow: '0 0 0 2px #93c5fd', transform: 'scale(1.02)' } : {})
                        }}
                      >
                        {word}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem', minWidth: 0 }}>
            {/* Статистика */}
            {objectiveStats.totalReference > 0 && (
              <div style={{
                background: 'white',
                borderRadius: '12px',
                padding: '0.875rem',
                border: '1px solid #e2e8f0'
              }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.6rem' }}>
                  <div style={{ textAlign: 'center', padding: '0.6rem', background: '#f8fafc', borderRadius: '8px' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#10b981' }}>{objectiveStats.correct}</div>
                    <div style={{ fontSize: '0.7rem', color: '#475569' }}>Правильно</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '0.6rem', background: '#f8fafc', borderRadius: '8px' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f59e0b' }}>{objectiveStats.error}</div>
                    <div style={{ fontSize: '0.7rem', color: '#475569' }}>С ошибкой</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '0.6rem', background: '#f8fafc', borderRadius: '8px' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#ef4444' }}>{objectiveStats.missed}</div>
                    <div style={{ fontSize: '0.7rem', color: '#475569' }}>Пропущено</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '0.6rem', background: '#f8fafc', borderRadius: '8px' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#8b5cf6' }}>{objectiveStats.extra}</div>
                    <div style={{ fontSize: '0.7rem', color: '#475569' }}>Лишние</div>
                  </div>
                </div>
                <div style={{ marginTop: '0.6rem', textAlign: 'center', padding: '0.5rem', background: '#eff6ff', borderRadius: '8px' }}>
                  <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#2563eb' }}>{objectiveStats.accuracy.toFixed(0)}%</div>
                  <div style={{ fontSize: '0.65rem', color: '#475569' }}>Точность</div>
                </div>
              </div>
            )}

            {/* Загрузка аудио */}
            <div
              onClick={() => !isProcessing && document.getElementById('file-input')?.click()}
              style={{
                border: `1.5px dashed ${isProcessing ? '#f59e0b' : file ? '#10b981' : '#cbd5e1'}`,
                borderRadius: '12px',
                padding: '1.25rem',
                textAlign: 'center',
                cursor: isProcessing ? 'default' : 'pointer',
                background: isProcessing ? '#fffbeb' : file ? '#ecfdf5' : 'white',
                transition: 'all 0.2s ease'
              }}
            >
              <input id="file-input" type="file" accept="audio/*" onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])} disabled={isProcessing} style={{ display: 'none' }} />

              {isProcessing ? (
                <>
                  <div style={{ display: 'inline-block', marginBottom: '0.5rem' }}>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ animation: 'spin 1s linear infinite' }}>
                      <circle cx="12" cy="12" r="10" stroke="#f59e0b" strokeWidth="2" strokeDasharray="30 30" strokeLinecap="round" fill="none" />
                    </svg>
                  </div>
                  <div style={{ fontSize: '0.85rem', fontWeight: '500', color: '#92400e' }}>{recognitionStatus}</div>
                  <div style={{ marginTop: '0.6rem' }}>
                    <div style={{ height: '4px', background: '#fde68a', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{ width: `${progress}%`, height: '100%', background: '#f59e0b', transition: 'width 0.3s' }} />
                    </div>
                  </div>
                  {totalSegments > 0 && (
                    <div style={{ fontSize: '0.7rem', color: '#b45309', marginTop: '0.4rem' }}>
                      {currentSegment} / {totalSegments}
                    </div>
                  )}
                </>
              ) : file ? (
                <>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="1.5" style={{ marginBottom: '0.4rem' }}>
                    <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" stroke="currentColor" fill="none" />
                    <path d="M7 9l5-5 5 5" stroke="currentColor" fill="none" strokeLinecap="round" />
                    <path d="M12 4v12" stroke="currentColor" fill="none" strokeLinecap="round" />
                  </svg>
                  <div style={{ fontSize: '0.85rem', fontWeight: '500', color: '#065f46' }}>{file.name}</div>
                  <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '0.25rem' }}>{audioDuration.toFixed(1)} сек</div>
                </>
              ) : (
                <>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5" style={{ marginBottom: '0.4rem' }}>
                    <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" stroke="currentColor" fill="none" />
                    <path d="M7 9l5-5 5 5" stroke="currentColor" fill="none" strokeLinecap="round" />
                    <path d="M12 4v12" stroke="currentColor" fill="none" strokeLinecap="round" />
                  </svg>
                  <div style={{ fontSize: '0.85rem', fontWeight: '500', color: '#475569' }}>Загрузить аудио</div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '0.25rem' }}>MP3, WAV, M4A</div>
                </>
              )}
            </div>

            {/* Аудио плеер */}
            {file && (
              <div style={{
                background: 'white',
                borderRadius: '12px',
                padding: '0.6rem',
                border: '1px solid #e2e8f0'
              }}>
                <audio
                  ref={audioRef}
                  controls
                  src={getAudioUrl()}
                  style={{ width: '100%', height: '44px' }}
                  onPlay={handleAudioPlay}
                  onPause={handleAudioPause}
                  onEnded={handleAudioEnded}
                  onLoadedMetadata={handleAudioLoadedMetadata}
                />
              </div>
            )}

            {/* Кнопка */}
            {file && (
              <button
                onClick={handleNewAttempt}
                disabled={isProcessing}
                style={{
                  width: '100%',
                  padding: '0.7rem',
                  background: isProcessing ? '#cbd5e1' : '#1e293b',
                  color: 'white',
                  border: 'none',
                  borderRadius: '10px',
                  fontSize: '0.85rem',
                  fontWeight: '600',
                  cursor: isProcessing ? 'not-allowed' : 'pointer',
                  transition: 'background 0.2s'
                }}
              >
                {isProcessing ? 'Обработка...' : '+ Новая проверка'}
              </button>
            )}
          </div>
        </div>

        {(!isProcessing && wordAnalysis.length > 0) && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', marginBottom: '1.25rem' }}>
            {/* Детальный анализ */}
            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '1rem',
              border: '1px solid #e2e8f0',
              minWidth: 0,
              overflow: 'hidden'
            }}>
              <div style={{ fontSize: '0.7rem', fontWeight: '600', color: '#475569', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Детальный анализ произношения
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '360px', overflowY: 'auto' }}>
                {wordAnalysis.map((item, i) => {
                  const expectedTimeSec = parseFloat(item.expectedTime);
                  const actualTimeSec = item.actualTime ? parseFloat(item.actualTime) : null;
                  const timeDiffSec = item.timeDiff ? parseFloat(item.timeDiff) : null;
                  return (
                    <div
                      key={i}
                      style={{
                        padding: '0.5rem 0.65rem',
                        background: '#f8fafc',
                        borderRadius: '8px',
                        borderLeft: `3px solid ${item.color}`
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.3rem', fontSize: '0.65rem', color: '#64748b' }}>
                        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                          <span style={{ fontWeight: '600' }}>#{item.position}</span>
                          <span>⏱️ {expectedTimeSec.toFixed(1)}с</span>
                          {actualTimeSec !== null && (
                            <span>→ {actualTimeSec.toFixed(1)}с</span>
                          )}
                          {timeDiffSec !== null && (
                            <span style={{ color: Math.abs(timeDiffSec) > 0.5 ? '#ef4444' : '#10b981' }}>
                              {timeDiffSec > 0 ? '+' : ''}{timeDiffSec.toFixed(1)}с
                            </span>
                          )}
                        </div>
                        <span style={{
                          fontWeight: '600',
                          padding: '0.1rem 0.4rem',
                          borderRadius: '4px',
                          background: `${item.color}15`,
                          color: item.color,
                          fontSize: '0.65rem'
                        }}>
                          {item.similarity.toFixed(0)}%
                        </span>
                      </div>
                      <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.8rem', flexWrap: 'wrap', alignItems: 'baseline' }}>
                        <span style={{ color: '#1e293b', fontWeight: '500' }}>📖 {item.expected}</span>
                        <span style={{ color: '#cbd5e1' }}>→</span>
                        <span style={{ color: '#f59e0b' }}>🎤 {item.actual}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Распознанный текст */}
            {transcription && (
              <div style={{
                background: 'white',
                borderRadius: '12px',
                padding: '1rem',
                border: '1px solid #e2e8f0',
                minWidth: 0,
                overflow: 'hidden'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.3rem' }}>
                  <div style={{ fontSize: '0.7rem', fontWeight: '600', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Распознанный текст
                  </div>
                  <div style={{ fontSize: '0.6rem', color: '#94a3b8', background: '#f1f5f9', padding: '0.15rem 0.4rem', borderRadius: '4px' }}>
                    {wordTimestamps.length} слов
                  </div>
                </div>
                <div style={{
                  fontSize: '0.85rem',
                  lineHeight: '1.5',
                  color: '#334155',
                  maxHeight: '360px',
                  overflowY: 'auto',
                  wordBreak: 'break-word',
                  whiteSpace: 'pre-wrap'
                }}>
                  {transcription}
                </div>
              </div>
            )}
          </div>
        )}
        {/* Debug лог */}
        <div style={{
          marginTop: '0.875rem',
          padding: '0.6rem 0.875rem',
          background: '#f1f5f9',
          borderRadius: '8px',
          fontSize: '0.7rem',
          fontFamily: 'monospace',
          color: '#64748b',
          maxHeight: '80px',
          overflowY: 'auto',
          border: '1px solid #e2e8f0'
        }}>
          {debugLog.map((log, i) => (
            <div key={i} style={{ padding: '0.15rem 0' }}>{log}</div>
          ))}
        </div>

        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    </div>
  );
}