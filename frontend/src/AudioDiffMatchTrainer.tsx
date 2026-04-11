import { useState, useRef, useEffect } from 'react';
import diff_match_patch from 'diff-match-patch';

type FinalKind = 'pending' | 'correct' | 'error' | 'missed';

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

  const CHUNK_DURATION = 10;       // длина чанка аудио (сек)
  const MAX_RECONNECT_ATTEMPTS = 5;
  const WS_URL = 'ws://localhost:2700';

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
    addDebugLog('Улучшение качества аудио для Vosk...');
    return applyNoiseReduction(audioBuffer);
  };

  // Обновление эталонного текста при его изменении
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

    const duration = recWords[recWords.length - 1].end || 0;

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

  /**
   * После buildDiffMatchAlignment дополняем expectedStart/expectedEnd для подсветки «дырок»
   */
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
      const wordsInGap = [];
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

  // Распознавание одного чанка с ретраями
  const processChunkWithRetry = async (
    chunk: any,
    chunkIndex: number,
    totalChunks: number,
    retryCount = 0
  ) => {
    return new Promise<any[]>((resolve) => {
      addDebugLog(`Чанк ${chunkIndex + 1}/${totalChunks} (попытка ${retryCount + 1})`);
      const ws = new WebSocket(WS_URL);
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
        // конфиг для Vosk: включаем разметку по словам
        ws.send(JSON.stringify({ config: { sample_rate: 16000, words: true, max_alternatives: 0 } }));
        setTimeout(() => {
          // берём моно-канал, конвертируем в 16-bit PCM и отправляем
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
    // при старте с начала — сбрасываем цвета
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
      setRecognitionStatus(`Распознавание завершено! ${uniqueResults.length} слов`);
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

  const renderLegend = () => (
    <div
      style={{
        display: 'flex',
        gap: '1.5rem',
        marginBottom: '1rem',
        padding: '0.5rem',
        background: '#f5f5f5',
        borderRadius: '8px',
        justifyContent: 'center',
        flexWrap: 'wrap'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: 20, background: COLORS.CURRENT, borderRadius: '4px' }} />
        <span>Сейчас (ожидаемый или распознанный интервал)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: 20, background: COLORS.CORRECT, borderRadius: '4px' }} />
        <span>Правильно (≥80%)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: 20, background: COLORS.ERROR, borderRadius: '4px' }} />
        <span>С ошибкой (40–80%)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: 20, background: COLORS.MISSED, borderRadius: '4px' }} />
        <span>Пропуск / не найдено</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: 20, background: COLORS.PENDING, borderRadius: '4px' }} />
        <span>Ещё не дошли</span>
      </div>
    </div>
  );

  const renderWordAnalysis = () => {
    if (!wordAnalysis.length) return null;

    return (
      <div
        style={{
          marginBottom: '2rem',
          padding: '1rem',
          background: '#f5f5f5',
          borderRadius: '12px'
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '1rem', fontSize: '1.1rem' }}>
          Детальный анализ произношения:
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
            maxHeight: '500px',
            overflowY: 'auto'
          }}
        >
          {wordAnalysis.map((item, i) => (
            <div
              key={i}
              style={{
                padding: '0.5rem',
                background: 'white',
                borderRadius: '6px',
                borderLeft: `5px solid ${item.color}`,
                display: 'grid',
                gridTemplateColumns: 'auto 1fr auto',
                gap: '1rem',
                alignItems: 'center'
              }}
            >
              <div style={{ fontWeight: 'bold', color: '#666' }}>{item.position}.</div>
              <div>
                <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem', color: '#666' }}>
                  <span>⏱️ exp:{item.expectedTime}с</span>
                  {item.actualTime && <span>🎤 act:{item.actualTime}с</span>}
                  {item.timeDiff && <span>📏 diff:{item.timeDiff}с</span>}
                </div>
                <div style={{ color: '#4caf50' }}>📖 {item.expected}</div>
                <div style={{ color: '#ff9800' }}>🎤 {item.actual}</div>
                {item.changes && item.changes.length > 0 && (
                  <div style={{ fontSize: '0.8rem', color: '#999', marginTop: '0.2rem' }}>
                    {item.changes.join(' • ')}
                  </div>
                )}
              </div>
              <div
                style={{
                  background: item.color,
                  color: 'white',
                  padding: '0.2rem 0.5rem',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  fontWeight: 'bold'
                }}
              >
                {item.similarity.toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ---------- Основной JSX ----------

  return (
    <div style={{ padding: '1rem', margin: '0 auto', maxWidth: '1200px' }}>
      <h1>Речевой тренажер (diff → match + стабильная подсветка)</h1>

      {renderLegend()}

      {/* Поле для ввода эталонного текста */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
          Эталонный текст:
        </label>
        <textarea
          value={referenceText}
          onChange={(e) => updateReferenceText(e.target.value)}
          disabled={isProcessing}
          style={{
            width: '100%',
            padding: '0.5rem',
            fontSize: '1rem',
            fontFamily: 'inherit',
            border: `2px solid ${isProcessing ? '#ccc' : '#4CAF50'}`,
            borderRadius: '8px',
            resize: 'vertical',
            minHeight: '120px',
            background: isProcessing ? '#f5f5f5' : 'white'
          }}
          placeholder="Введите эталонный текст здесь..."
        />
        <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.3rem' }}>
          Количество слов: {referenceWordsRef.current.length}
        </div>
      </div>

      <div
        style={{
          marginBottom: '1rem',
          padding: '0.5rem',
          background: '#2d2d2d',
          color: '#f0f0f0',
          borderRadius: '6px',
          fontSize: '0.8rem',
          maxHeight: '200px',
          overflowY: 'auto',
          fontFamily: 'monospace'
        }}
      >
        {debugLog.map((log, index) => (
          <div
            key={index}
            style={{
              padding: '0.1rem 0',
              borderBottom: index < debugLog.length - 1 ? '1px solid #444' : 'none'
            }}
          >
            {log}
          </div>
        ))}
      </div>

      {objectiveStats.totalReference > 0 && (
        <div
          style={{
            marginBottom: '1rem',
            padding: '1rem',
            background: '#e8eaf6',
            borderRadius: '8px'
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Статистика:</div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
              gap: '1rem'
            }}
          >
            <div style={{ color: COLORS.CORRECT }}>Правильно: {objectiveStats.correct}</div>
            <div style={{ color: COLORS.ERROR }}>С ошибкой: {objectiveStats.error}</div>
            <div style={{ color: COLORS.MISSED }}>Пропущено: {objectiveStats.missed}</div>
            <div style={{ color: '#ff8f00' }}>Лишние: {objectiveStats.extra}</div>
            <div style={{ color: '#00796b', fontWeight: 'bold' }}>
              Точность: {objectiveStats.accuracy.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {isProcessing && (
        <div style={{ marginBottom: '1rem', padding: '1rem', background: '#f0f0f0', borderRadius: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span>{recognitionStatus}</span>
            <span>{progress}%</span>
          </div>
          <div style={{ height: '10px', background: '#e0e0e0', borderRadius: '5px', overflow: 'hidden' }}>
            <div
              style={{
                height: '100%',
                background: '#4CAF50',
                width: `${progress}%`,
                transition: 'width 0.3s ease'
              }}
            />
          </div>
          {totalSegments > 0 && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#666' }}>
              Чанк {currentSegment} из {totalSegments} • Распознано слов: {allResultsRef.current.length}
            </div>
          )}
        </div>
      )}

      {partial && isProcessing && (
        <div
          style={{
            marginBottom: '1rem',
            padding: '0.5rem',
            background: '#fff3e0',
            borderRadius: '6px',
            fontStyle: 'italic',
            color: '#ff9800'
          }}
        >
          Текущий фрагмент: "{partial}"
        </div>
      )}

      {/* Зона загрузки аудио */}
      <div
        style={{
          border: `3px ${isProcessing ? 'solid' : 'dashed'} ${isProcessing ? '#ff9800' : '#4CAF50'}`,
          padding: '1rem',
          borderRadius: '12px',
          textAlign: 'center',
          cursor: isProcessing ? 'wait' : 'pointer',
          background: isProcessing ? '#fff3e0' : '#e8f5e8',
          marginBottom: '2rem'
        }}
        onClick={() => !isProcessing && document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept="audio/*,audio/wav,audio/mp3,audio/m4a"
          onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
          disabled={isProcessing}
          style={{ display: 'none' }}
        />

        {isProcessing ? (
          <div>
            <div style={{ fontSize: '3rem', animation: 'spin 1s linear infinite' }}>。</div>
            <div style={{ fontSize: '1.2rem', marginTop: '1rem' }}>{recognitionStatus}</div>
            {totalSegments > 0 && (
              <div style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.5rem' }}>
                Чанк {currentSegment} из {totalSegments}
              </div>
            )}
          </div>
        ) : file ? (
          <div>
            <div style={{ fontSize: '3rem' }}>✓</div>
            <div style={{ fontSize: '1.2rem', color: '#2e7d32', marginTop: '0.5rem' }}>{file.name}</div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '4rem' }}>+</div>
            <div style={{ fontSize: '1.3rem', marginTop: '1rem' }}>Загрузите аудиофайл</div>
          </div>
        )}
      </div>

      {file && (
        <div style={{ marginBottom: '2rem', padding: '0.5rem', background: '#f0f0f0', borderRadius: '12px' }}>
          <div style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>
            Аудио: <strong>{file.name}</strong>
            {audioDuration > 0 && (
              <span style={{ marginLeft: '1rem', fontSize: '0.9rem', color: '#666' }}>
                {audioDuration.toFixed(1)} сек
              </span>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <audio
              key={audioUrlRef.current || 'audio'}
              ref={audioRef}
              controls
              src={getAudioUrl()}
              style={{ width: '100%', maxWidth: '800px' }}
              preload="metadata"
              onPlay={handleAudioPlay}
              onPause={handleAudioPause}
              onEnded={handleAudioEnded}
              onLoadedMetadata={handleAudioLoadedMetadata}
            />
          </div>
        </div>
      )}

      <div style={{ marginBottom: '2rem', padding: '1rem', background: '#e3f2fd', borderRadius: '12px' }}>
        <div style={{ fontSize: '1rem', marginBottom: '1rem', color: '#1976d2' }}>
          Эталонный текст ({referenceWordsRef.current.length} слов):
        </div>
        <div style={{ fontSize: '1.1rem', lineHeight: '1.6', display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
          {referenceWordsRef.current.map((word, index) => {
            const color = wordColors[index] || COLORS.PENDING;

            let tooltip = '';
            if (color === COLORS.CURRENT) tooltip = 'Сейчас (интервал слова)';
            else if (color === COLORS.CORRECT) tooltip = 'Произнесено правильно';
            else if (color === COLORS.ERROR) tooltip = 'Произнесено с ошибкой';
            else if (color === COLORS.MISSED) tooltip = 'Пропущено или не сопоставлено';
            else tooltip = 'Ожидание';

            return (
              <span
                key={index}
                style={{
                  color:
                    color === COLORS.CORRECT
                      ? '#2e7d32'
                      : color === COLORS.ERROR
                        ? '#e65100'
                        : color === COLORS.MISSED
                          ? '#c62828'
                          : color === COLORS.CURRENT
                            ? '#0d47a1'
                            : '#757575',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '4px',
                  fontWeight: color === COLORS.CURRENT ? '700' : '400',
                  background:
                    color === COLORS.CORRECT
                      ? '#c8e6c9'
                      : color === COLORS.ERROR
                        ? '#fff3e0'
                        : color === COLORS.MISSED
                          ? '#ffcdd2'
                          : color === COLORS.CURRENT
                            ? '#bbdefb'
                            : '#f5f5f5',
                  boxShadow: color === COLORS.CURRENT ? '0 0 10px rgba(33,150,243,0.3)' : 'none',
                  display: 'inline-block',
                  marginBottom: '0.2rem',
                  cursor: 'help'
                }}
                title={`${tooltip}: ${word}`}
              >
                {word}
              </span>
            );
          })}
        </div>
      </div>

      {/* Детальный анализ */}
      {!isProcessing && wordAnalysis.length > 0 && renderWordAnalysis()}

      {/* Распознанный текст целиком */}
      {transcription && (
        <div style={{ marginBottom: '2rem', padding: '1rem', background: '#f3e5f5', borderRadius: '12px' }}>
          <div style={{ fontSize: '1rem', marginBottom: '0.5rem', fontWeight: 'bold' }}>
            Распознанный текст ({wordTimestamps.length} слов):
          </div>
          <div
            style={{
              fontSize: '1rem',
              lineHeight: '1.5',
              padding: '0.5rem',
              background: '#f8f8f8',
              borderRadius: '6px',
              whiteSpace: 'pre-wrap'
            }}
          >
            {transcription}
          </div>
        </div>
      )}

      {/* Кнопка сброса */}
      {file && (
        <button
          onClick={handleNewAttempt}
          disabled={isProcessing}
          style={{
            padding: '0.8rem 1.5rem',
            background: isProcessing ? '#ccc' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
            fontSize: '1rem',
            width: '100%',
            fontWeight: 'bold'
          }}
        >
          {isProcessing ? 'Отмена...' : 'Новая проверка'}
        </button>
      )}

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}