import { useState, useRef, useEffect } from 'react';
import diff_match_patch from 'diff-match-patch';

export default function AudioFileRecognizer() {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [partial, setPartial] = useState('');
  const [wordColors, setWordColors] = useState([]);
  const [wordTimestamps, setWordTimestamps] = useState([]);
  const [currentPlayingWordIndex, setCurrentPlayingWordIndex] = useState(-1);
  const [recognitionStatus, setRecognitionStatus] = useState('');
  const [debugLog, setDebugLog] = useState([]);
  const [audioDuration, setAudioDuration] = useState(0);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);
  const [isAudioReady, setIsAudioReady] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentSegment, setCurrentSegment] = useState(0);
  const [totalSegments, setTotalSegments] = useState(0);
  const [wordAnalysis, setWordAnalysis] = useState([]);

  const [objectiveStats, setObjectiveStats] = useState({
    correct: 0,
    error: 0,
    missed: 0,
    extra: 0,
    totalReference: 0,
    totalRecognized: 0,
    accuracy: 0
  });

  const [sentences, setSentences] = useState([]);

  const wsRef = useRef(null);
  const audioRef = useRef(null);
  const animationFrameRef = useRef(null);
  const isHighlightingRef = useRef(false);
  const wordMappingRef = useRef([]);
  const audioUrlRef = useRef(null);
  const isCancelledRef = useRef(false);
  const audioContextRef = useRef(null);
  const allResultsRef = useRef([]);
  const audioBufferRef = useRef(null);
  const dmpRef = useRef(null);
  const wordColorsRef = useRef([]);
  const lastUpdateTimeRef = useRef(0);

  const referenceText = 'Уважаемые преподаватели, коллеги! ' +
    'Меня зовут _ _. Моё исследование посвящено автоматическому определению букв по произнесенным словам с использованием интеллектуальных средств обработки звука. ' +
    'Актуальность: Эта задача критически важна в телекоммуникациях и системах сигнализации, где требуется высокая точность даже в шумных условиях и при различиях в произношении. ' +
    'Цель: Разработать и экспериментально проверить методику анализа звуковых сигналов для идентификации букв, используя современные алгоритмы.';

  const referenceWordsRef = useRef([]);

  const CHUNK_DURATION = 10;
  const MAX_RECONNECT_ATTEMPTS = 5;
  const WS_URL = 'ws://localhost:2700';
  const TIME_WINDOW = 5.0;

  const COLORS = {
    CURRENT: '#2196f3',
    CORRECT: '#4caf50',
    ERROR: '#ff9800',
    MISSED: '#f44336',
    PENDING: '#9e9e9e'
  };

  const THRESHOLDS = {
    CORRECT: 80,
    ERROR: 40,
    MISSED: 0
  };

  useEffect(() => {
    dmpRef.current = new diff_match_patch();
    dmpRef.current.Diff_Timeout = 1.5;
    dmpRef.current.Match_Threshold = 0.4;
    dmpRef.current.Match_Distance = 500;

    const sentencesArray = segmentTextBySentences(referenceText);
    setSentences(sentencesArray);

    const allWords = [];
    sentencesArray.forEach(sentence => {
      sentence.words.forEach(word => allWords.push(word));
    });

    referenceWordsRef.current = allWords;

    const initialColors = Array(allWords.length).fill(COLORS.PENDING);
    setWordColors(initialColors);
    wordColorsRef.current = initialColors;

    addDebugLog(`📚 Текст разбит на ${sentencesArray.length} предложений (${allWords.length} слов)`);
  }, []);

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      isCancelledRef.current = true;
    };
  }, []);

  const addDebugLog = (message) => {
    console.log(message);
    setDebugLog(prev => [...prev.slice(-30), `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const normalizeWord = (word) => {
    if (!word) return '';
    return word.toLowerCase()
      .replace(/[.,!?;:()\[\]{}"'-]/g, '')
      .replace(/ё/g, 'е')
      .trim();
  };

  const calculateSimilarity = (word1, word2) => {
    if (!word1 || !word2) return { similarity: 0, changes: [] };

    const norm1 = normalizeWord(word1);
    const norm2 = normalizeWord(word2);

    const diffs = dmpRef.current.diff_main(norm1, norm2);
    dmpRef.current.diff_cleanupSemantic(diffs);

    let equalLength = 0;
    let changes = [];

    diffs.forEach(([op, text]) => {
      if (op === 0) {
        equalLength += text.length;
      } else if (op === -1) {
        changes.push(`❌ удалено: "${text}"`);
      } else if (op === 1) {
        changes.push(`➕ вставлено: "${text}"`);
      }
    });

    const totalLength = norm1.length + norm2.length;
    const similarity = totalLength > 0 ? (equalLength * 2) / totalLength * 100 : 0;

    return { similarity, changes };
  };

  const segmentTextBySentences = (text) => {
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
      const words = sentence.split(' ').filter(w => w.length > 0);
      return {
        id: index,
        text: sentence.trim(),
        words: words,
        expectedDuration: words.length * 0.4 + 0.5
      };
    });
  };

  const chunkArray = (array, size) => {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  };

  const splitAudioIntoChunks = (audioBuffer, chunkDuration, overlapDuration = 1.0) => {
    const chunks = [];
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
        startSample: startSample,
        endSample: endSample,
        startTime: startSample / sampleRate,
        endTime: endSample / sampleRate,
        duration: actualChunkSamples / sampleRate,
        buffer: extractAudioSegment(audioBuffer, startSample, endSample)
      });

      startSample += stepSamples;
      chunkIndex++;
    }

    addDebugLog(`📦 Аудио разбито на ${chunks.length} чанков по ${chunkDuration}с`);
    return chunks;
  };

  const extractAudioSegment = (audioBuffer, startSample, endSample) => {
    const length = endSample - startSample;
    if (length <= 0) return null;

    const segmentBuffer = audioContextRef.current.createBuffer(
      audioBuffer.numberOfChannels,
      length,
      audioBuffer.sampleRate
    );

    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel);
      const segmentChannelData = segmentBuffer.getChannelData(channel);

      for (let i = 0; i < length; i++) {
        if (startSample + i < channelData.length) {
          segmentChannelData[i] = channelData[startSample + i];
        }
      }
    }

    return segmentBuffer;
  };

  const processChunkWithRetry = async (chunk, chunkIndex, totalChunks, retryCount = 0) => {
    return new Promise((resolve) => {
      addDebugLog(`🔄 Чанк ${chunkIndex + 1}/${totalChunks} (попытка ${retryCount + 1})`);

      const ws = new WebSocket(WS_URL);
      let chunkResults = [];
      let complete = false;

      const timeoutId = setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) ws.close();

        if (!complete && retryCount < MAX_RECONNECT_ATTEMPTS) {
          setTimeout(async () => {
            const retryResults = await processChunkWithRetry(chunk, chunkIndex, totalChunks, retryCount + 1);
            resolve(retryResults);
          }, 1000);
        } else {
          resolve(chunkResults);
        }
      }, chunk.duration * 1000 + 10000);

      ws.onopen = () => {
        ws.send(JSON.stringify({
          config: {
            sample_rate: 16000,
            words: true,
            max_alternatives: 0
          }
        }));

        setTimeout(() => {
          const channelData = chunk.buffer.getChannelData(0);
          const pcm16 = new Int16Array(channelData.length);

          for (let i = 0; i < channelData.length; i++) {
            pcm16[i] = Math.max(-32768, Math.min(32767, channelData[i] * 32767));
          }

          const pcmData = new Uint8Array(pcm16.buffer);
          ws.send(pcmData);

          setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send('{"eof" : 1}');
            }
          }, 500);
        }, 200);
      };

      ws.onmessage = (e) => {
        try {
          let data;
          if (typeof e.data === 'string') {
            data = JSON.parse(e.data);
          } else {
            data = JSON.parse(new TextDecoder().decode(e.data));
          }

          if (data.result) {
            data.result.forEach(word => {
              chunkResults.push({
                word: word.word,
                start: (word.start || 0) + chunk.startTime,
                end: (word.end || 0) + chunk.startTime,
                confidence: word.confidence || 1.0,
                chunkIndex: chunkIndex
              });
            });
          }

          if (data.text && !complete) {
            complete = true;
            clearTimeout(timeoutId);
            ws.close();

            setProgress(Math.floor(((chunkIndex + 1) / totalChunks) * 100));
            setCurrentSegment(chunkIndex + 1);

            resolve(chunkResults);
          }
        } catch (err) { }
      };

      ws.onerror = () => {
        if (!complete && retryCount < MAX_RECONNECT_ATTEMPTS) {
          clearTimeout(timeoutId);
          ws.close();

          setTimeout(async () => {
            const retryResults = await processChunkWithRetry(chunk, chunkIndex, totalChunks, retryCount + 1);
            resolve(retryResults);
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

  const processAllChunks = async (chunks) => {
    const allResults = [];
    const totalChunks = chunks.length;

    setTotalSegments(totalChunks);

    for (let i = 0; i < chunks.length; i++) {
      if (isCancelledRef.current) break;

      const chunkResults = await processChunkWithRetry(chunks[i], i, totalChunks);
      allResults.push(...chunkResults);

      allResultsRef.current = allResults;

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    return allResults;
  };

  const deduplicateResults = (results, overlapThreshold = 0.5) => {
    if (results.length === 0) return results;

    const sorted = [...results].sort((a, b) => a.start - b.start);
    const unique = [];
    const used = new Set();

    for (let i = 0; i < sorted.length; i++) {
      const current = sorted[i];
      let isDuplicate = false;

      for (let j = 0; j < unique.length; j++) {
        const existing = unique[j];

        if (normalizeWord(current.word) === normalizeWord(existing.word) &&
          Math.abs(current.start - existing.start) < overlapThreshold) {
          isDuplicate = true;
          break;
        }
      }

      if (!isDuplicate && !used.has(i)) {
        unique.push(current);
        used.add(i);
      }
    }

    addDebugLog(`🧹 Удаление дубликатов: было ${results.length}, стало ${unique.length}`);
    return unique;
  };

  const createWorkingMapping = (referenceWords, recognizedWords, duration) => {
    addDebugLog(`🕐 Создание time-based mapping с окном ${TIME_WINDOW}с`);

    const mapping = [];
    const usedRecognized = new Set();
    const sortedRecognized = [...recognizedWords].sort((a, b) => a.start - b.start);

    const expectedTimes = referenceWords.map((_, idx) => (idx / referenceWords.length) * duration);

    for (let refIdx = 0; refIdx < referenceWords.length; refIdx++) {
      const expectedTime = expectedTimes[refIdx];

      const candidates = sortedRecognized.filter(rec =>
        !usedRecognized.has(rec) &&
        Math.abs(rec.start - expectedTime) <= TIME_WINDOW
      );

      if (candidates.length > 0) {
        let bestMatch = null;
        let bestScore = -1;

        candidates.forEach(candidate => {
          const { similarity } = calculateSimilarity(referenceWords[refIdx], candidate.word);
          const timeDiff = Math.abs(candidate.start - expectedTime);
          const timeScore = 100 - (timeDiff / TIME_WINDOW) * 50;
          // 70% лингвистика, 30% время
          const totalScore = similarity * 0.7 + timeScore * 0.3;

          if (totalScore > bestScore) {
            bestScore = totalScore;
            bestMatch = candidate;
          }
        });

        if (bestMatch) {
          mapping.push({
            referenceIndex: refIdx,
            referenceWord: referenceWords[refIdx],
            voskWord: bestMatch,
            expectedTime,
            actualTime: bestMatch.start,
            timeDiff: Math.abs(bestMatch.start - expectedTime)
          });
          usedRecognized.add(bestMatch);
        }
      }
    }

    for (let refIdx = 0; refIdx < referenceWords.length; refIdx++) {
      const existing = mapping.find(m => m.referenceIndex === refIdx);
      if (!existing) {
        mapping.push({
          referenceIndex: refIdx,
          referenceWord: referenceWords[refIdx],
          voskWord: null,
          expectedTime: expectedTimes[refIdx],
          actualTime: null,
          timeDiff: null
        });
      }
    }

    const remainingRecognized = sortedRecognized.filter(rec => !usedRecognized.has(rec));
    remainingRecognized.forEach(rec => {
      mapping.push({
        referenceIndex: -1,
        referenceWord: '',
        voskWord: rec,
        isExtra: true
      });
    });

    return mapping;
  };

  const analyzeAllWords = (referenceWords, recognizedWords, mapping, duration) => {
    console.clear();
    console.log('%c' + '='.repeat(100), 'color: #2196f3');
    console.log('%c🎯 АНАЛИЗ ПРОИЗНОШЕНИЯ (TIME-BASED)', 'color: #2196f3; font-size: 16px; font-weight: bold');
    console.log('%c' + '='.repeat(100), 'color: #2196f3');

    const analysis = [];
    const enhancedMapping = [];
    let stats = { correct: 0, error: 0, missed: 0, extra: 0 };

    mapping.forEach(item => {
      if (item.voskWord && item.referenceWord) {
        const { similarity, changes } = calculateSimilarity(
          item.referenceWord,
          item.voskWord.word
        );

        let status;
        if (similarity >= THRESHOLDS.CORRECT) {
          status = 'correct';
          stats.correct++;
        } else if (similarity >= THRESHOLDS.ERROR) {
          status = 'error';
          stats.error++;
        } else {
          status = 'missed';
          stats.missed++;
        }

        enhancedMapping.push({
          ...item,
          similarity,
          status,
          changes
        });
      } else if (item.isExtra) {
        stats.extra++;
        enhancedMapping.push({
          ...item,
          similarity: 0,
          status: 'extra',
          changes: []
        });
      } else {
        stats.missed++;
        enhancedMapping.push({
          ...item,
          similarity: 0,
          status: 'missed',
          changes: []
        });
      }
    });

    stats.totalReference = referenceWords.length;
    stats.totalRecognized = recognizedWords.length;
    stats.accuracy = (stats.correct / referenceWords.length) * 100;
    setObjectiveStats(stats);

    const sortedMapping = enhancedMapping
      .filter(m => m.referenceIndex >= 0)
      .sort((a, b) => a.referenceIndex - b.referenceIndex);

    sortedMapping.forEach(item => {
      if (!item.voskWord) {
        console.log('%c' + '-'.repeat(80), 'color: #f44336');
        console.log(`%c📌 СЛОВО ${item.referenceIndex + 1} [exp:${item.expectedTime.toFixed(1)}с] (ПРОПУЩЕНО)`, 'color: #f44336; font-weight: bold');
        console.log(`%c📖 Эталон: "${item.referenceWord}"`, 'color: #f44336');

        analysis.push({
          position: item.referenceIndex + 1,
          expectedTime: item.expectedTime.toFixed(1),
          expected: item.referenceWord,
          actual: '(не распознано)',
          similarity: 0,
          status: 'missed',
          color: COLORS.MISSED
        });
      } else {
        const statusColor = item.similarity >= THRESHOLDS.CORRECT ? '#4caf50' :
          item.similarity >= THRESHOLDS.ERROR ? '#ff9800' : '#f44336';
        const statusText = item.similarity >= THRESHOLDS.CORRECT ? 'ПРАВИЛЬНО' :
          item.similarity >= THRESHOLDS.ERROR ? 'С ОШИБКОЙ' : 'ДРУГОЕ СЛОВО';

        console.log('%c' + '-'.repeat(80), `color: ${statusColor}`);
        console.log(`%c📌 СЛОВО ${item.referenceIndex + 1} [exp:${item.expectedTime.toFixed(1)}с | act:${item.actualTime.toFixed(1)}с | diff:${item.timeDiff.toFixed(1)}с]`, `color: ${statusColor}; font-weight: bold`);
        console.log(`%c📖 Эталон:        "${item.referenceWord}"`, 'color: #4caf50');
        console.log(`%c🎤 Распознано:    "${item.voskWord.word}"`, 'color: #ff9800');
        console.log(`%c🔍 Сходство:      ${item.similarity.toFixed(1)}%`, `color: ${statusColor}; font-weight: bold`);
        console.log(`%c🏷️ Статус:        ${statusText}`, `color: ${statusColor}; font-weight: bold`);

        if (item.changes && item.changes.length > 0) {
          console.log('%c✏️ Изменения:', 'color: #ff9800');
          item.changes.forEach(change => console.log(`   ${change}`));
        }

        analysis.push({
          position: item.referenceIndex + 1,
          expectedTime: item.expectedTime.toFixed(1),
          actualTime: item.actualTime.toFixed(1),
          timeDiff: item.timeDiff.toFixed(1),
          expected: item.referenceWord,
          actual: item.voskWord.word,
          similarity: item.similarity,
          status: item.status,
          color: statusColor,
          changes: item.changes
        });
      }
    });

    console.log('%c' + '='.repeat(100), 'color: #2196f3');
    console.log('%c📊 ИТОГОВАЯ СТАТИСТИКА', 'color: #2196f3; font-size: 14px; font-weight: bold');
    console.log(`%c✅ Правильно: ${stats.correct} слов`, 'color: #4caf50');
    console.log(`%c🟠 С ошибкой: ${stats.error} слов`, 'color: #ff9800');
    console.log(`%c❌ Пропущено: ${stats.missed} слов`, 'color: #f44336');
    console.log(`%c➕ Лишние: ${stats.extra} слов`, 'color: #ff8f00');

    setWordAnalysis(analysis);
    return enhancedMapping;
  };

  // Подсветка
  const updateWordHighlighting = (currentTime) => {
    if (!wordMappingRef.current.length) return;

    const newColors = [...wordColorsRef.current];
    let currentWordIndex = -1;

    const wordsWithTime = wordMappingRef.current
      .filter(item => item.voskWord && item.referenceIndex >= 0)
      .map(item => ({
        refIndex: item.referenceIndex,
        start: item.voskWord.start,
        end: item.voskWord.end,
        status: item.status
      }));

    const currentWord = wordsWithTime.find(word =>
      currentTime >= word.start && currentTime <= word.end
    );

    if (currentWord) {
      currentWordIndex = currentWord.refIndex;
    }

    for (let i = 0; i < referenceWordsRef.current.length; i++) {
      const wordWithTime = wordsWithTime.find(w => w.refIndex === i);

      if (wordWithTime) {
        if (currentWord && currentWord.refIndex === i) {
          newColors[i] = COLORS.CURRENT;
        } else if (wordWithTime.end <= currentTime) {
          switch (wordWithTime.status) {
            case 'correct':
              newColors[i] = COLORS.CORRECT;
              break;
            case 'error':
              newColors[i] = COLORS.ERROR;
              break;
            default:
              newColors[i] = COLORS.MISSED;
          }
        } else {
          newColors[i] = COLORS.PENDING;
        }
      } else {
        const expectedTime = (i / referenceWordsRef.current.length) * audioDuration;
        if (currentTime > expectedTime + 1.5) {
          newColors[i] = COLORS.MISSED;
        } else {
          newColors[i] = COLORS.PENDING;
        }
      }
    }

    let colorsChanged = false;
    for (let i = 0; i < newColors.length; i++) {
      if (newColors[i] !== wordColorsRef.current[i]) {
        colorsChanged = true;
        break;
      }
    }

    if (colorsChanged) {
      setWordColors(newColors);
      wordColorsRef.current = newColors;
    }

    setCurrentPlayingWordIndex(currentWordIndex);
    setAudioCurrentTime(currentTime);
  };

  const startHighlightingLoop = () => {
    if (isHighlightingRef.current) return;
    if (!audioRef.current || audioRef.current.paused) return;

    isHighlightingRef.current = true;
    let lastUpdate = 0;

    const highlightLoop = () => {
      if (!audioRef.current || !isHighlightingRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        return;
      }

      const audio = audioRef.current;
      if (audio.paused) {
        isHighlightingRef.current = false;
        cancelAnimationFrame(animationFrameRef.current);
        return;
      }

      const now = Date.now();
      if (now - lastUpdate > 50) {
        updateWordHighlighting(audio.currentTime);
        lastUpdate = now;
      }

      animationFrameRef.current = requestAnimationFrame(highlightLoop);
    };

    animationFrameRef.current = requestAnimationFrame(highlightLoop);
  };

  const handleAudioPlay = () => {
    if (audioRef.current.currentTime < 0.5) {
      const resetColors = Array(referenceWordsRef.current.length).fill(COLORS.PENDING);
      setWordColors(resetColors);
      wordColorsRef.current = resetColors;
      setCurrentPlayingWordIndex(-1);
    }
    startHighlightingLoop();
  };

  const handleAudioPause = () => {
    isHighlightingRef.current = false;
  };

  const handleAudioEnded = () => {
    isHighlightingRef.current = false;

    if (wordMappingRef.current.length) {
      const finalColors = wordMappingRef.current.map(item => {
        if (item.referenceIndex < 0) return null;

        switch (item.status) {
          case 'correct':
            return COLORS.CORRECT;
          case 'error':
            return COLORS.ERROR;
          default:
            return COLORS.MISSED;
        }
      }).filter(color => color !== null);

      setWordColors(finalColors);
      wordColorsRef.current = finalColors;
    }
  };

  const handleAudioLoadedMetadata = () => {
    const audio = audioRef.current;
    if (!audio) return;
    setAudioDuration(audio.duration || 0);
    setIsAudioReady(true);
  };

  const processFile = async (selectedFile) => {
    console.clear();
    addDebugLog('🎯 Начало обработки файла (time-based)');

    isCancelledRef.current = false;
    allResultsRef.current = [];

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

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

    wordMappingRef.current = [];
    isHighlightingRef.current = false;
    wordColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS.PENDING);

    audioUrlRef.current = URL.createObjectURL(selectedFile);

    try {
      setRecognitionStatus('Чтение и декодирование аудио...');
      const arrayBuffer = await readFileAsArrayBuffer(selectedFile);

      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      audioBufferRef.current = audioBuffer;

      const duration = audioBuffer.duration;
      addDebugLog(`📊 Аудио: ${duration.toFixed(2)} секунд`);
      setAudioDuration(duration);

      setRecognitionStatus('Разбиение аудио на чанки...');
      const chunks = splitAudioIntoChunks(audioBuffer, CHUNK_DURATION, 1.0);

      setRecognitionStatus('Распознавание речи...');
      const rawResults = await processAllChunks(chunks);

      if (isCancelledRef.current) {
        addDebugLog('❌ Обработка отменена');
        setIsProcessing(false);
        return;
      }

      const uniqueResults = deduplicateResults(rawResults);

      setRecognitionStatus('Сопоставление по времени...');
      const mapping = createWorkingMapping(
        referenceWordsRef.current,
        uniqueResults,
        duration
      );

      const enhancedMapping = analyzeAllWords(
        referenceWordsRef.current,
        uniqueResults,
        mapping,
        duration
      );

      wordMappingRef.current = enhancedMapping;

      setWordTimestamps(uniqueResults);
      setTranscription(uniqueResults.map(w => w.word).join(' '));

      setProgress(100);
      setIsProcessing(false);
      setRecognitionStatus(`Распознавание завершено! ${uniqueResults.length} слов`);

    } catch (error) {
      addDebugLog(`❌ Критическая ошибка: ${error.message}`);
      console.error(error);
      setRecognitionStatus(`Ошибка: ${error.message}`);
      setIsProcessing(false);
    }
  };

  const readFileAsArrayBuffer = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Ошибка чтения файла'));
      reader.readAsArrayBuffer(file);
    });
  };

  const handleNewAttempt = () => {
    isCancelledRef.current = true;
    isHighlightingRef.current = false;

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }

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
      correct: 0, error: 0, missed: 0, extra: 0, totalReference: 0, totalRecognized: 0, accuracy: 0
    });

    wordMappingRef.current = [];
    allResultsRef.current = [];
    audioBufferRef.current = null;
    wordColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS.PENDING);

    addDebugLog('🔄 Новая попытка');
  };

  const getAudioUrl = () => {
    return audioUrlRef.current || '';
  };

  const renderLegend = () => (
    <div style={{
      display: 'flex',
      gap: '1.5rem',
      marginBottom: '1rem',
      padding: '0.5rem',
      background: '#f5f5f5',
      borderRadius: '8px',
      justifyContent: 'center',
      flexWrap: 'wrap'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: '20px', background: COLORS.CURRENT, borderRadius: '4px' }}></div>
        <span>Текущее слово</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: '20px', background: COLORS.CORRECT, borderRadius: '4px' }}></div>
        <span>Произнесено правильно (≥80%)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: '20px', background: COLORS.ERROR, borderRadius: '4px' }}></div>
        <span>С ошибкой (40-80%)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: '20px', background: COLORS.MISSED, borderRadius: '4px' }}></div>
        <span>Пропущено/другое ({"<"}40%)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '20px', height: '20px', background: COLORS.PENDING, borderRadius: '4px' }}></div>
        <span>Ожидание</span>
      </div>
    </div>
  );

  const renderWordAnalysis = () => {
    if (wordAnalysis.length === 0) return null;

    return (
      <div style={{
        marginBottom: '2rem',
        padding: '1rem',
        background: '#f5f5f5',
        borderRadius: '12px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '1rem', fontSize: '1.1rem' }}>
          📊 Детальный анализ произношения:
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
          maxHeight: '500px',
          overflowY: 'auto'
        }}>
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
              <div style={{ fontWeight: 'bold', color: '#666' }}>
                {item.position}.
              </div>
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
              <div style={{
                background: item.color,
                color: 'white',
                padding: '0.2rem 0.5rem',
                borderRadius: '4px',
                fontSize: '0.8rem',
                fontWeight: 'bold'
              }}>
                {item.similarity.toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: '1rem', margin: '0 auto', maxWidth: '1200px' }}>
      <h1>🎤 Речевой тренажер (time-based)</h1>

      {renderLegend()}

      {objectiveStats.totalReference > 0 && (
        <div style={{
          marginBottom: '1rem',
          padding: '1rem',
          background: '#e8eaf6',
          borderRadius: '8px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>
            📊 Статистика:
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
            gap: '1rem'
          }}>
            <div style={{ color: COLORS.CORRECT }}>
              ✅ Правильно: {objectiveStats.correct}
            </div>
            <div style={{ color: COLORS.ERROR }}>
              🟠 С ошибкой: {objectiveStats.error}
            </div>
            <div style={{ color: COLORS.MISSED }}>
              ❌ Пропущено: {objectiveStats.missed}
            </div>
            <div style={{ color: '#ff8f00' }}>
              ➕ Лишние: {objectiveStats.extra}
            </div>
            <div style={{ color: '#00796b', fontWeight: 'bold' }}>
              🎯 Точность: {objectiveStats.accuracy.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {isProcessing && (
        <div style={{
          marginBottom: '1rem',
          padding: '1rem',
          background: '#f0f0f0',
          borderRadius: '8px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span>{recognitionStatus}</span>
            <span>{progress}%</span>
          </div>
          <div style={{
            height: '10px',
            background: '#e0e0e0',
            borderRadius: '5px',
            overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              background: '#4CAF50',
              width: `${progress}%`,
              transition: 'width 0.3s ease'
            }} />
          </div>
        </div>
      )}

      {partial && isProcessing && (
        <div style={{
          marginBottom: '1rem',
          padding: '0.5rem',
          background: '#fff3e0',
          borderRadius: '6px',
          fontStyle: 'italic',
          color: '#ff9800'
        }}>
          ⏳ Текущий фрагмент: "{partial}"
        </div>
      )}

      <div style={{
        border: `3px ${isProcessing ? 'solid' : 'dashed'} ${isProcessing ? '#ff9800' : '#4CAF50'}`,
        padding: '1rem',
        borderRadius: '12px',
        textAlign: 'center',
        cursor: isProcessing ? 'wait' : 'pointer',
        background: isProcessing ? '#fff3e0' : '#e8f5e8',
        marginBottom: '2rem'
      }}
        onClick={() => !isProcessing && document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept="audio/*,audio/wav,audio/mp3,audio/m4a"
          onChange={(e) => e.target.files[0] && processFile(e.target.files[0])}
          disabled={isProcessing}
          style={{ display: 'none' }}
        />

        {isProcessing ? (
          <div>
            <div style={{ fontSize: '3rem', animation: 'spin 1s linear infinite' }}>。</div>
            <div style={{ fontSize: '1.2rem', marginTop: '1rem' }}>
              {recognitionStatus}
            </div>
          </div>
        ) : file ? (
          <div>
            <div style={{ fontSize: '3rem' }}>✓</div>
            <div style={{ fontSize: '1.2rem', color: '#2e7d32', marginTop: '0.5rem' }}>
              {file.name}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '4rem' }}>+</div>
            <div style={{ fontSize: '1.3rem', marginTop: '1rem' }}>
              Загрузите аудиофайл
            </div>
          </div>
        )}
      </div>

      {file && (
        <div style={{
          marginBottom: '2rem',
          padding: '0.5rem',
          background: '#f0f0f0',
          borderRadius: '12px'
        }}>
          <div style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>
            🔊 <strong>{file.name}</strong>
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

      <div style={{
        marginBottom: '2rem',
        padding: '1rem',
        background: '#e3f2fd',
        borderRadius: '12px',
      }}>
        <div style={{ fontSize: '1rem', marginBottom: '1rem', color: '#1976d2' }}>
          Эталонный текст ({referenceWordsRef.current.length} слов):
        </div>

        <div style={{
          fontSize: '1.1rem',
          lineHeight: '1.6',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.3rem'
        }}>
          {referenceWordsRef.current.map((word, index) => {
            const color = wordColors[index] || COLORS.PENDING;

            let tooltip = '';
            if (color === COLORS.CURRENT) tooltip = 'Сейчас произносится';
            else if (color === COLORS.CORRECT) tooltip = 'Произнесено правильно';
            else if (color === COLORS.ERROR) tooltip = 'Произнесено с ошибкой';
            else if (color === COLORS.MISSED) tooltip = 'Пропущено или заменено';
            else tooltip = 'Ожидание';

            return (
              <span
                key={index}
                style={{
                  color: color === COLORS.CORRECT ? '#2e7d32' :
                    color === COLORS.ERROR ? '#e65100' :
                      color === COLORS.MISSED ? '#c62828' :
                        color === COLORS.CURRENT ? '#0d47a1' : '#757575',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '4px',
                  fontWeight: color === COLORS.CURRENT ? '700' : '400',
                  background: color === COLORS.CORRECT ? '#c8e6c9' :
                    color === COLORS.ERROR ? '#fff3e0' :
                      color === COLORS.MISSED ? '#ffcdd2' :
                        color === COLORS.CURRENT ? '#bbdefb' : '#f5f5f5',
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

      {!isProcessing && wordAnalysis.length > 0 && renderWordAnalysis()}

      {transcription && (
        <div style={{
          marginBottom: '2rem',
          padding: '1rem',
          background: '#f3e5f5',
          borderRadius: '12px'
        }}>
          <div style={{ fontSize: '1rem', marginBottom: '0.5rem', fontWeight: 'bold' }}>
            Распознанный текст ({wordTimestamps.length} слов):
          </div>
          <div style={{
            fontSize: '1rem',
            lineHeight: '1.5',
            padding: '0.5rem',
            background: '#f8f8f8',
            borderRadius: '6px',
            whiteSpace: 'pre-wrap'
          }}>
            {transcription}
          </div>
        </div>
      )}

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