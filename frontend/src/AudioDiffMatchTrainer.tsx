import { useState, useRef, useEffect } from 'react';
import diff_match_patch from 'diff-match-patch';

type FinalKind = 'pending' | 'correct' | 'error' | 'missed';
type ModelType = 'vosk' | 'whisper' | 'wav2vec2';

// ==================== Цветовая схема ====================
const COLORS_UI = {
  primary: '#2969e3',
  primaryLight: '#3f81fd33',
  primaryHover: '#2a72f8',
  primaryDark: '#1549ab',
  secondary: '#528eff',
  secondaryLight: '#528eff8f',

  bgMain: '#FFFFFF',
  bgCard: '#FFFFFF',
  bgSurface: '#f7f9fb',
  bgHover: '#f2f5f8',
  bgCardHover: '#f2f5f8f5',

  border: '#e8eef2',
  borderDark: '#d5dfe6',
  borderFocus: '#2969e3',

  textPrimary: '#13181bcc',
  textSecondary: '#13181b8f',
  textTertiary: '#13181b47',
  textInverse: '#FFFFFF',
  textAccent: '#2969e3',

  shadow: '#060a0c05',
  shadowHover: '#060a0c0f',
  shadowCard: '#060a0c1f',
  shadowFloat: '#060a0c1f',

  success: '#10b981',
  successLight: '#10b98115',
  warning: '#f59e0b',
  warningLight: '#f59e0b15',
  error: '#ef4444',
  errorLight: '#ef444415',
  info: '#2969e3',

  correct: '#4caf50',
  errorWord: '#ff9800',
  missed: '#f44336',
  current: '#8b5cf6',
  pending: '#d5dfe6',
  good: '#2969e3',

  // Дополнительные
  overlayLight: '#f2f5f88f',
  overlayMedium: '#f2f5f847',
  overlayDark: '#f2f5f8cc',
  glow: '#3f81fd',
};

// ==================== ИПТ для ОТДЕЛЬНОГО СЛОВА ====================
interface WordIPT {
  wordIndex: number;
  word: string;
  startTime: number;
  endTime: number;

  totalLetters: number;
  recognizedLetters: number;
  correctLetters: number;

  precision: number;
  recall: number;
  fScore: number;
  confidence: number;

  errors: number;
  substitutions: number;
  insertions: number;
  deletions: number;

  ipt: number;
  iptNormalized: number;

  status: 'perfect' | 'good' | 'medium' | 'poor' | 'critical';
}

interface AggregateIPT {
  words: WordIPT[];
  weightedAverageIPT: number;
  weightedAverageNormalized: number;
  simpleAverageIPT: number;
  perfectCount: number;
  goodCount: number;
  mediumCount: number;
  poorCount: number;
  criticalCount: number;
  totalErrors: number;
  totalLetters: number;
  totalCorrectLetters: number;
}

// Функция для очистки слова от знаков препинания
const cleanWordForAnalysis = (word: string): string => {
  if (!word) return '';
  return word
    .trim()
    .replace(/[.,!?;:()\[\]{}"'-]/g, '')
    .replace(/\s+/g, '');
};

const computeWordIPT = (
  expectedWord: string,
  recognizedWord: string | null,
  confidence: number,
  lambda: number = 0.05
): Omit<WordIPT, 'wordIndex' | 'word' | 'startTime' | 'endTime'> => {

  // Очищаем слова от знаков препинания
  const cleanExpected = cleanWordForAnalysis(expectedWord);

  if (!recognizedWord) {
    const totalLetters = cleanExpected.length;
    return {
      totalLetters,
      recognizedLetters: 0,
      correctLetters: 0,
      precision: 0,
      recall: 0,
      fScore: 0,
      confidence: 0,
      errors: totalLetters,
      substitutions: 0,
      insertions: 0,
      deletions: totalLetters,
      ipt: -lambda * totalLetters,
      iptNormalized: 0,
      status: 'critical'
    };
  }

  const cleanRecognized = cleanWordForAnalysis(recognizedWord);

  const normExpected = cleanExpected.toLowerCase();
  const normRecognized = cleanRecognized.toLowerCase();

  const expectedLetters = normExpected.split('');
  const recognizedLetters = normRecognized.split('');

  const N = expectedLetters.length;
  const M = recognizedLetters.length;

  const dp: { correct: number; score: number }[][] = Array(N + 1).fill(null).map(
    () => Array(M + 1).fill(null).map(() => ({ correct: 0, score: 0 }))
  );

  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= M; j++) {
      if (i === 0 && j === 0) {
        dp[i][j] = { correct: 0, score: 0 };
      } else if (i === 0) {
        dp[i][j] = { correct: 0, score: dp[i][j - 1].score - 1 };
      } else if (j === 0) {
        dp[i][j] = { correct: 0, score: dp[i - 1][j].score - 1 };
      } else {
        const matchCost = expectedLetters[i - 1] === recognizedLetters[j - 1] ? 1 : -1;
        const matchScore = dp[i - 1][j - 1].score + matchCost;
        const matchCorrect = dp[i - 1][j - 1].correct + (matchCost === 1 ? 1 : 0);

        const insertScore = dp[i][j - 1].score - 1;
        const deleteScore = dp[i - 1][j].score - 1;

        if (matchScore >= insertScore && matchScore >= deleteScore) {
          dp[i][j] = { correct: matchCorrect, score: matchScore };
        } else if (insertScore >= deleteScore) {
          dp[i][j] = { correct: dp[i][j - 1].correct, score: insertScore };
        } else {
          dp[i][j] = { correct: dp[i - 1][j].correct, score: deleteScore };
        }
      }
    }
  }

  const K = dp[N][M].correct;

  let substitutions = 0, insertions = 0, deletions = 0;
  let i = N, j = M;

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && expectedLetters[i - 1] === recognizedLetters[j - 1]) {
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1].score >= dp[i - 1][j].score)) {
      insertions++;
      j--;
    } else if (i > 0 && (j === 0 || dp[i - 1][j].score > dp[i][j - 1].score)) {
      deletions++;
      i--;
    } else {
      substitutions++;
      i--; j--;
    }
  }

  const totalErrors = substitutions + insertions + deletions;

  const precision = M > 0 ? K / M : 0;
  const recall = N > 0 ? K / N : 0;
  const fScore = (precision + recall) / 2;

  let ipt = fScore * confidence - lambda * totalErrors;
  let iptNormalized = Math.max(0, Math.min(100, ((ipt + 0.5) / 1.5) * 100));
  if (ipt < 0) iptNormalized = 0;

  let status: WordIPT['status'];
  if (ipt >= 0.8) status = 'perfect';
  else if (ipt >= 0.5) status = 'good';
  else if (ipt >= 0.3) status = 'medium';
  else if (ipt >= 0.2) status = 'poor';
  else status = 'critical';

  return {
    totalLetters: N,
    recognizedLetters: M,
    correctLetters: K,
    precision,
    recall,
    fScore,
    confidence,
    errors: totalErrors,
    substitutions,
    insertions,
    deletions,
    ipt,
    iptNormalized,
    status
  };
};

const computeAggregateIPT = (mapping: any[], lambda: number = 0.05): AggregateIPT => {
  const words: WordIPT[] = [];
  let perfectCount = 0, goodCount = 0, mediumCount = 0, poorCount = 0, criticalCount = 0;
  let totalErrors = 0, totalLetters = 0, totalCorrectLetters = 0;
  let weightedSumIPT = 0;
  let totalDuration = 0;

  for (let i = 0; i < mapping.length; i++) {
    const item = mapping[i];
    const expectedWord = item.referenceWord;
    const recognizedWord = item.voskWord?.word || null;
    const confidence = item.voskWord?.confidence || 0.85;
    const startTime = item.expectedStart ?? item.expectedTime ?? 0;
    const endTime = item.expectedEnd ?? (item.expectedTime ?? 0) + 0.4;
    const duration = endTime - startTime;

    const wordIPTData = computeWordIPT(expectedWord, recognizedWord, confidence, lambda);

    const wordIPT: WordIPT = {
      wordIndex: i,
      word: expectedWord,
      startTime,
      endTime,
      ...wordIPTData
    };

    words.push(wordIPT);

    if (wordIPT.status === 'perfect') perfectCount++;
    else if (wordIPT.status === 'good') goodCount++;
    else if (wordIPT.status === 'medium') mediumCount++;
    else if (wordIPT.status === 'poor') poorCount++;
    else criticalCount++;

    totalErrors += wordIPT.errors;
    totalLetters += wordIPT.totalLetters;
    totalCorrectLetters += wordIPT.correctLetters;

    weightedSumIPT += wordIPT.ipt * duration;
    totalDuration += duration;
  }

  const weightedAverageIPT = totalDuration > 0 ? weightedSumIPT / totalDuration : 0;
  const weightedAverageNormalized = Math.max(0, Math.min(100, ((weightedAverageIPT + 0.5) / 1.5) * 100));
  const simpleAverageIPT = words.length > 0 ? words.reduce((sum, w) => sum + w.ipt, 0) / words.length : 0;

  return {
    words,
    weightedAverageIPT,
    weightedAverageNormalized,
    simpleAverageIPT,
    perfectCount,
    goodCount,
    mediumCount,
    poorCount,
    criticalCount,
    totalErrors,
    totalLetters,
    totalCorrectLetters
  };
};

// Компонент для отображения графика ИПТ по словам
const IPTChart = ({ words, currentTime, onWordClick, audioDuration, statsCounts }: {
  words: WordIPT[];
  currentTime: number;
  onWordClick: (startTime: number) => void;
  audioDuration: number;
  statsCounts: { perfect: number; good: number; medium: number; poor: number; critical: number };
}) => {
  if (words.length === 0) return null;

  const maxIPT = Math.max(...words.map(w => w.ipt), 0.5);
  const height = 100;

  const getBarColor = (status: WordIPT['status']) => {
    switch (status) {
      case 'perfect': return COLORS_UI.success;
      case 'good': return COLORS_UI.primary;
      case 'medium': return COLORS_UI.warning;
      case 'poor': return COLORS_UI.error;
      case 'critical': return '#dc2626';
    }
  };

  const getStatusLabel = (status: WordIPT['status']) => {
    switch (status) {
      case 'perfect': return 'Отлично';
      case 'good': return 'Хорошо';
      case 'medium': return 'Средне';
      case 'poor': return 'Плохо';
      case 'critical': return 'Критично';
    }
  };

  const lastWordEnd = words[words.length - 1]?.endTime || audioDuration;
  const effectiveDuration = lastWordEnd;

  const isBeyondLastWord = currentTime > lastWordEnd + 0.1;
  const currentPositionPercent = !isBeyondLastWord && effectiveDuration > 0
    ? (currentTime / effectiveDuration) * 100
    : 100;

  const avgIPT = words.reduce((sum, w) => sum + w.ipt, 0) / words.length;

  const barsWithPositions = words.map((word, idx) => {
    let x1 = (word.startTime / effectiveDuration) * 100;
    let x2 = (word.endTime / effectiveDuration) * 100;

    const MIN_WIDTH_PERCENT = 0.5;
    const width = x2 - x1;

    if (width < MIN_WIDTH_PERCENT) {
      const center = (x1 + x2) / 2;
      x1 = Math.max(0, center - MIN_WIDTH_PERCENT / 2);
      x2 = Math.min(100, center + MIN_WIDTH_PERCENT / 2);
    }

    return {
      ...word,
      idx,
      x1,
      x2,
      width: x2 - x1,
      normalizedIPT: Math.max(0.05, Math.min(1, word.ipt / maxIPT)),
      y: height - (Math.max(0.05, Math.min(1, word.ipt / maxIPT)) * height)
    };
  });

  const sortedBars = [...barsWithPositions].sort((a, b) => b.normalizedIPT - a.normalizedIPT);

  return (
    <div style={{
      background: COLORS_UI.bgCard,
      borderRadius: '20px',
      padding: '1.5rem',
      border: `1px solid ${COLORS_UI.border}`,
      boxShadow: `0 4px 12px ${COLORS_UI.shadow}`,
      marginBottom: '1.5rem',
      transition: 'all 0.2s ease'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
          <span style={{ fontSize: '0.875rem', fontWeight: '600', color: COLORS_UI.textSecondary, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Пословный ИПТ
          </span>
          <span style={{
            fontSize: '1.25rem',
            fontWeight: '700',
            color: avgIPT >= 0.8 ? COLORS_UI.success :
              avgIPT >= 0.5 ? COLORS_UI.primary :
                avgIPT >= 0.3 ? COLORS_UI.warning :
                  COLORS_UI.error
          }}>
            {avgIPT.toFixed(3)}
          </span>
          {/* <span style={{
            fontSize: '0.75rem',
            padding: '0.25rem 0.75rem',
            borderRadius: '20px',
            background: `${avgIPT >= 0.8 ? COLORS_UI.success :
              avgIPT >= 0.5 ? COLORS_UI.primary :
                avgIPT >= 0.3 ? COLORS_UI.warning :
                  COLORS_UI.error}15`,
            color: avgIPT >= 0.8 ? COLORS_UI.success :
              avgIPT >= 0.5 ? COLORS_UI.primary :
                avgIPT >= 0.3 ? COLORS_UI.warning :
                  COLORS_UI.error,
            fontWeight: '500'
          }}>
            {((avgIPT + 0.5) / 1.5 * 100).toFixed(0)}%
          </span> */}
        </div>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          {(['perfect', 'good', 'medium', 'poor', 'critical'] as const).map(status => {
            const counts = { perfect: statsCounts.perfect, good: statsCounts.good, medium: statsCounts.medium, poor: statsCounts.poor, critical: statsCounts.critical };
            return (
              <div key={status} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: getBarColor(status) }} />
                <span style={{ fontSize: '0.75rem', color: COLORS_UI.textSecondary }}>{getStatusLabel(status)}</span>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: COLORS_UI.textPrimary }}>{counts[status]}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ position: 'relative', marginBottom: '0.5rem' }}>
        <svg width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" style={{ display: 'block' }}>
          <line x1="0" y1={height} x2="100" y2={height} stroke={COLORS_UI.borderDark} strokeWidth="0.5" />

          {sortedBars.map((bar) => {
            const needsStroke = bar.width < 1.0;

            return (
              <rect
                key={`bar-${bar.idx}`}
                x={`${bar.x1}%`}
                y={bar.y}
                width={`${Math.max(0.3, bar.width)}%`}
                height={bar.normalizedIPT * height}
                fill={getBarColor(bar.status)}
                stroke={needsStroke ? getBarColor(bar.status) : 'none'}
                strokeWidth={needsStroke ? "0.5" : "0"}
                opacity={currentTime >= bar.startTime && currentTime <= bar.endTime ? 1 : 0.85}
                style={{
                  cursor: 'pointer',
                  transition: 'opacity 0.2s',
                  pointerEvents: 'visiblePainted'
                }}
                onClick={() => onWordClick(bar.startTime)}
                onMouseEnter={(e) => {
                  e.currentTarget.style.opacity = '1';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.opacity = currentTime >= bar.startTime && currentTime <= bar.endTime ? '1' : '0.85';
                }}
              />
            );
          })}

          {audioDuration > lastWordEnd + 0.1 && (
            <rect
              x={`${(lastWordEnd / effectiveDuration) * 100}%`}
              y="0"
              width={`${Math.min(100 - (lastWordEnd / effectiveDuration) * 100, ((audioDuration - lastWordEnd) / effectiveDuration) * 100)}%`}
              height={height}
              fill={COLORS_UI.bgSurface}
              opacity={0.5}
              style={{ pointerEvents: 'none' }}
            />
          )}
        </svg>

        {currentTime > 0 && effectiveDuration > 0 && !isBeyondLastWord && (
          <div style={{
            position: 'absolute',
            bottom: 0,
            left: `${currentPositionPercent}%`,
            width: '2px',
            height: `${height}px`,
            background: COLORS_UI.textPrimary,
            transform: 'translateX(-50%)',
            pointerEvents: 'none',
            zIndex: 20,
            transition: 'left 0.05s linear'
          }}>
            <div style={{
              position: 'absolute',
              bottom: -28,
              left: -12,
              fontSize: '0.7rem',
              background: COLORS_UI.primary,
              color: COLORS_UI.textInverse,
              padding: '0.25rem 0.6rem',
              borderRadius: '12px',
              whiteSpace: 'nowrap',
              fontWeight: 500,
              boxShadow: `0 2px 4px ${COLORS_UI.shadow}`
            }}>
              {currentTime.toFixed(1)}с
            </div>
          </div>
        )}

        {isBeyondLastWord && currentTime > lastWordEnd + 0.5 && (
          <div style={{
            position: 'absolute',
            bottom: -28,
            right: 0,
            fontSize: '0.7rem',
            color: COLORS_UI.textTertiary,
            background: COLORS_UI.bgSurface,
            padding: '0.2rem 0.5rem',
            borderRadius: '12px',
            fontStyle: 'italic'
          }}>
            тишина
          </div>
        )}
      </div>

      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginTop: '0.5rem',
        fontSize: '0.7rem',
        color: COLORS_UI.textTertiary
      }}>
        <span>0:00</span>
        <span>{Math.floor(effectiveDuration / 60)}:{Math.floor(effectiveDuration % 60).toString().padStart(2, '0')}</span>
        {audioDuration > effectiveDuration + 0.1 && (
          <span style={{ color: COLORS_UI.textSecondary }}>
            + тишина {((audioDuration - effectiveDuration).toFixed(1))}с
          </span>
        )}
      </div>
    </div>
  );
};

// Компонент для отображения детального списка слов с ИПТ
const WordIPTList = ({ words, currentTime, onWordClick, wordTimestamps }: {
  words: WordIPT[];
  currentTime: number;
  onWordClick: (startTime: number) => void;
  wordTimestamps: any[];
}) => {
  if (words.length === 0) return null;

  const getStatusColor = (status: WordIPT['status']) => {
    switch (status) {
      case 'perfect': return COLORS_UI.success;
      case 'good': return COLORS_UI.primary;
      case 'medium': return COLORS_UI.warning;
      case 'poor': return COLORS_UI.error;
      case 'critical': return '#dc2626';
    }
  };

  const cleanWord = (word: string) => {
    return word.replace(/[.,!?;:()\[\]{}"'-]/g, '');
  };

  const getStatusIcon = (status: WordIPT['status']) => {
    switch (status) {
      case 'perfect': return '✓✓';
      case 'good': return '✓';
      case 'medium': return '○';
      case 'poor': return '△';
      case 'critical': return '✗';
      default: return '•';
    }
  };

  // Получаем распознанное слово для индекса
  const getRecognizedWord = (wordIndex: number) => {
    const word = words[wordIndex];
    if (!word) return null;
    // Ищем соответствующее слово в wordTimestamps по времени
    const recognized = wordTimestamps.find(w =>
      Math.abs(w.start - word.startTime) < 0.1 &&
      Math.abs(w.end - word.endTime) < 0.1
    );
    return recognized ? cleanWord(recognized.word) : '(не распознано)';
  };

  return (
    <div style={{
      background: COLORS_UI.bgCard,
      borderRadius: '20px',
      padding: '1.5rem',
      border: `1px solid ${COLORS_UI.border}`,
      boxShadow: `0 4px 12px ${COLORS_UI.shadow}`,
      minWidth: 0,
      overflow: 'hidden',
      transition: 'all 0.2s ease',
      height: 'fit-content'
    }}>
      <div style={{ fontSize: '0.875rem', fontWeight: '600', color: COLORS_UI.textSecondary, marginBottom: '1.25rem', textTransform: 'uppercase', letterSpacing: '0.5px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontSize: '1.1rem' }}>📋</span>
        Детальный разбор слов
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '460px', overflowY: 'auto' }}>
        {words.map((word) => {
          const recognizedWord = getRecognizedWord(word.wordIndex);
          return (
            <div
              key={word.wordIndex}
              onClick={() => onWordClick(word.startTime)}
              style={{
                padding: '0.875rem 1rem',
                background: currentTime >= word.startTime && currentTime <= word.endTime ? COLORS_UI.primaryLight : COLORS_UI.bgSurface,
                borderRadius: '16px',
                borderLeft: `4px solid ${getStatusColor(word.status)}`,
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                ...(currentTime >= word.startTime && currentTime <= word.endTime ? {
                  boxShadow: `0 4px 12px ${COLORS_UI.shadowHover}`,
                  transform: 'translateX(4px)'
                } : {})
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <span style={{
                    fontWeight: '600',
                    fontSize: '1rem',
                    color: COLORS_UI.textPrimary,
                    background: COLORS_UI.bgCard,
                    padding: '0.25rem 0.5rem',
                    borderRadius: '8px'
                  }}>{cleanWord(word.word)}</span>
                  <span style={{ color: COLORS_UI.textTertiary, fontSize: '0.9rem' }}>→</span>
                  <span style={{
                    fontWeight: '500',
                    fontSize: '1rem',
                    color: recognizedWord === '(не распознано)' ? COLORS_UI.error : COLORS_UI.warning
                  }}>{recognizedWord}</span>
                  <span style={{ fontSize: '0.75rem', color: COLORS_UI.textTertiary, fontFamily: 'monospace' }}>
                    {word.startTime.toFixed(1)}с – {word.endTime.toFixed(1)}с
                  </span>
                </div>
                <div style={{
                  padding: '0.25rem 0.875rem',
                  borderRadius: '20px',
                  background: `${getStatusColor(word.status)}15`,
                  fontSize: '0.75rem',
                  fontWeight: '600',
                  color: getStatusColor(word.status),
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.375rem'
                }}>
                  <span>{getStatusIcon(word.status)}</span>
                  <span>ИПТ: {word.ipt.toFixed(3)}</span>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '1.25rem', fontSize: '0.75rem', color: COLORS_UI.textSecondary, flexWrap: 'wrap' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ fontWeight: '600' }}>📖</span> {word.totalLetters} букв
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ fontWeight: '600' }}>✓</span> {word.correctLetters}/{word.recognizedLetters}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: word.errors > 0 ? COLORS_UI.error : COLORS_UI.textTertiary }}>
                  <span style={{ fontWeight: '600' }}>⚠️</span> {word.errors} ошибок
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ fontWeight: '600' }}>🎯</span> {(word.precision * 100).toFixed(0)}%
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ fontWeight: '600' }}>🔮</span> {(word.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Компонент для отображения распознанного текста
const RecognizedTextBlock = ({ transcription, wordTimestamps, onWordClick }: {
  transcription: string;
  wordTimestamps: any[];
  onWordClick: (startTime: number) => void;
}) => {
  if (!transcription && wordTimestamps.length === 0) return null;

  const cleanWord = (word: string) => {
    return word.replace(/[.,!?;:()\[\]{}"'-]/g, '');
  };

  return (
    <div style={{
      background: COLORS_UI.bgCard,
      borderRadius: '20px',
      padding: '1.5rem',
      border: `1px solid ${COLORS_UI.border}`,
      boxShadow: `0 4px 12px ${COLORS_UI.shadow}`,
      transition: 'all 0.2s ease',
      height: 'fit-content'
    }}>
      <div style={{
        fontSize: '0.875rem',
        fontWeight: '600',
        color: COLORS_UI.textSecondary,
        marginBottom: '1rem',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }}>
        <span style={{ fontSize: '1.1rem' }}>🎤</span>
        Распознанный текст
        <span style={{
          marginLeft: 'auto',
          fontSize: '0.7rem',
          padding: '0.2rem 0.6rem',
          borderRadius: '12px',
          background: COLORS_UI.bgSurface,
          color: COLORS_UI.textTertiary
        }}>
          {wordTimestamps.length} слов
        </span>
      </div>

      <div style={{
        background: COLORS_UI.bgSurface,
        borderRadius: '16px',
        padding: '1rem',
        maxHeight: '400px',
        overflowY: 'auto',
        fontSize: '1rem',
        lineHeight: '1.6',
        color: COLORS_UI.textPrimary
      }}>
        {wordTimestamps.length > 0 ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {wordTimestamps.map((word, idx) => (
              <span
                key={idx}
                onClick={() => onWordClick(word.start)}
                style={{
                  cursor: 'pointer',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '8px',
                  background: COLORS_UI.bgCard,
                  border: `1px solid ${COLORS_UI.border}`,
                  transition: 'all 0.2s ease',
                  fontSize: '0.95rem'
                }}
              >
                {cleanWord(word.word)}
              </span>
            ))}
          </div>
        ) : (
          <div style={{ color: COLORS_UI.textTertiary, textAlign: 'center', padding: '1rem' }}>
            Распознанный текст появится здесь после обработки аудио
          </div>
        )}
      </div>

      {transcription && (
        <div style={{
          marginTop: '0.75rem',
          fontSize: '0.7rem',
          color: COLORS_UI.textTertiary,
          textAlign: 'right',
          fontFamily: 'monospace'
        }}>
          Полный текст: {transcription.length} символов
        </div>
      )}
    </div>
  );
};

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

  const [aggregateIPT, setAggregateIPT] = useState<AggregateIPT | null>(null);

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
  const finalStateRef = useRef<string[]>([]);
  // Храним цвета на основе ИПТ для каждого слова
  const iptColorsRef = useRef<string[]>([]);

  const referenceWordsRef = useRef<string[]>([]);     // все слова эталона по порядку
  const [selectedModel, setSelectedModel] = useState<ModelType>('vosk');

  const CHUNK_DURATION = 10;       // длина чанка аудио (сек)
  const MAX_RECONNECT_ATTEMPTS = 5;

  const COLORS_WORD = {
    CURRENT: COLORS_UI.current,
    CORRECT: COLORS_UI.correct,
    GOOD: COLORS_UI.good,
    ERROR: COLORS_UI.errorWord,
    MISSED: COLORS_UI.missed,
    PENDING: COLORS_UI.pending
  };

  const THRESHOLDS = {
    CORRECT: 80,
    ERROR: 40
  };

  const TIME_WINDOW_LOCAL = 2.0;   // локальное окно по времени для улучшения match
  const TIME_WINDOW_GLOBAL = 6.0;  // максимальный допустимый сдвиг по времени для глобального match

  const WORD_RE = /\p{L}+/u;       // "слово" = последовательность букв (любой алфавит)
  const SIM_THRESHOLD_LOCAL = 60;
  const SIM_THRESHOLD_GLOBAL = 70;

  const UPDATE_INTERVAL_MS = 50;

  const cleanWord = (word: string) => {
    if (!word) return '';
    return word
      .replace(/[.,!?;:()\[\]{}"'-]/g, '')
      .trim();
  };

  // Шумоподавление (только для Vosk)
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
    highPass.frequency.value = 50;
    highPass.Q.value = 0.5;

    const lowPass = offlineContext.createBiquadFilter();
    lowPass.type = 'lowpass';
    lowPass.frequency.value = 12000;
    lowPass.Q.value = 0.5;

    const noiseGate = offlineContext.createGain();

    const compressor = offlineContext.createDynamicsCompressor();
    compressor.threshold.value = -16;
    compressor.knee.value = 6;
    compressor.ratio.value = 2;
    compressor.attack.value = 0.01;
    compressor.release.value = 0.15;

    const gain = offlineContext.createGain();
    gain.gain.value = 1.05;

    source.connect(highPass);
    highPass.connect(lowPass);
    lowPass.connect(compressor);
    compressor.connect(noiseGate);
    noiseGate.connect(gain);
    gain.connect(offlineContext.destination);

    source.start();
    const renderedBuffer = await offlineContext.startRendering();
    addDebugLog('Шумоподавление применено');
    return renderedBuffer;
  };

  const enhanceAudioForVosk = async (audioBuffer: AudioBuffer): Promise<AudioBuffer> => {
    // Whisper и Wav2Vec2 не требуют шумоподавления
    if (selectedModel === 'whisper' || selectedModel === 'wav2vec2') {
      addDebugLog(`${selectedModel.toUpperCase()} не требует дополнительного шумоподавления`);
      return audioBuffer;
    }

    addDebugLog('Улучшение качества аудио для Vosk...');
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

    // Изначально все слова — в состоянии "ожидание"
    const initialColors = Array(allWords.length).fill(COLORS_WORD.PENDING);
    setWordColors(initialColors);
    wordColorsRef.current = initialColors;
    finalStateRef.current = Array(allWords.length).fill('pending');
    iptColorsRef.current = Array(allWords.length).fill(COLORS_WORD.PENDING);
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
    return cleanWordForAnalysis(word).toLowerCase();
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
      const expectedTime = recWords[approxIdx]?.start ?? (duration ? (i / refTokens.length) * duration : 0);
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
  // но ограничиваемся окном по времени, чтобы "Слово1" не приклеилось к позднему "Слово1".
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

  // Функция для получения цвета на основе статуса ИПТ
  const getIPTColorForWord = (status: WordIPT['status']): string => {
    switch (status) {
      case 'perfect': return COLORS_WORD.CORRECT;
      case 'good': return COLORS_WORD.GOOD;
      case 'medium': return COLORS_WORD.ERROR;
      case 'poor': return COLORS_WORD.MISSED;
      case 'critical': return COLORS_WORD.MISSED;
      default: return COLORS_WORD.PENDING;
    }
  };

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
    const improvedLocal = improveWithMatchLocal(referenceWords, recWords, refToRecIdx, evalRecIdx, dmp);
    const finalRefToRec = globalFallbackMatch(referenceWords, recWords, improvedLocal.refToRecIdx, dmp);
    const duration = recWords.length ? recWords[recWords.length - 1].end : 0;
    const mapping: any[] = [];
    const analysis: any[] = [];
    let correct = 0, error = 0, missed = 0;
    let lastRecIdx = -1;  // предотвращаем "откат" по индексам Vosk
    let lastTime = 0;     // предотвращаем "откат" по ожидаемому времени

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
        const expectedBase = duration ? (i / referenceWords.length) * duration : 0;
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
          expected: cleanWord(refWord),
          actual: '(не распознано)',
          similarity: 0,
          status: 'missed',
          color: COLORS_WORD.MISSED,
          changes: []
        });
        lastTime = expectedTime;
        missed++;
      } else {
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
          color = COLORS_WORD.CORRECT;
          correct++;
        } else if (sim >= THRESHOLDS.ERROR) {
          status = 'error';
          color = COLORS_WORD.ERROR;
          error++;
        } else {
          status = 'missed';
          color = COLORS_WORD.MISSED;
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
          expected: cleanWord(refWord),
          actual: cleanWord(voskWord.word),
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

    const aggregate = computeAggregateIPT(mapping, 0.05);
    setAggregateIPT(aggregate);

    // Заполняем iptColorsRef на основе ИПТ
    const newIptColors = Array(referenceWords.length).fill(COLORS_WORD.PENDING);
    for (let i = 0; i < aggregate.words.length; i++) {
      const wordIPT = aggregate.words[i];
      newIptColors[i] = getIPTColorForWord(wordIPT.status);
      addDebugLog(`Слово "${wordIPT.word}" (индекс ${i}) статус ИПТ: ${wordIPT.status} → цвет: ${newIptColors[i] === COLORS_WORD.CORRECT ? 'зеленый' : newIptColors[i] === COLORS_WORD.ERROR ? 'оранжевый' : newIptColors[i] === COLORS_WORD.MISSED ? 'красный' : 'серый'}`);
    }
    iptColorsRef.current = newIptColors;

    addDebugLog(`ИПТ = ${aggregate.weightedAverageIPT.toFixed(3)} (${aggregate.weightedAverageNormalized.toFixed(1)}%) | Слов: ${aggregate.words.length}`);

    return mapping;
  };

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

  const splitAudioIntoChunks = (audioBuffer: AudioBuffer, chunkDuration: number, overlapDuration = 1.0) => {
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

  const getWebSocketUrl = () => {
    if (selectedModel === 'whisper') {
      return 'ws://localhost:2701';
    } else if (selectedModel === 'wav2vec2') {
      return 'ws://localhost:2702';
    } else {
      return 'ws://localhost:2700';
    }
  };

  // Для Whisper отправляем весь файл целиком
  const processFullAudioWhisper = async (audioBuffer: AudioBuffer): Promise<any[]> => {
    return new Promise((resolve) => {
      addDebugLog(`Отправка полного аудио в Whisper (${audioBuffer.duration.toFixed(1)}с)`);

      const ws = new WebSocket(getWebSocketUrl());
      let results: any[] = [];

      const timeoutId = setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) ws.close();
        addDebugLog('Таймаут Whisper, возвращаем то что есть');
        resolve(results);
      }, audioBuffer.duration * 1000 + 30000); // +30 секунд на обработку

      ws.onopen = () => {
        addDebugLog('WebSocket Whisper открыт, отправляем конфигурацию');
        ws.send(JSON.stringify({ config: { sample_rate: 16000 } }));

        // Конвертируем весь AudioBuffer в PCM16 и отправляем
        setTimeout(() => {
          const channelData = audioBuffer.getChannelData(0);
          const pcm16 = new Int16Array(channelData.length);
          for (let i = 0; i < channelData.length; i++) {
            pcm16[i] = Math.max(-32768, Math.min(32767, channelData[i] * 32767));
          }

          addDebugLog(`Отправка ${pcm16.length} сэмплов аудио`);
          ws.send(new Uint8Array(pcm16.buffer));

          setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
              addDebugLog('Отправка EOF в Whisper');
              ws.send('{"eof" : 1}');
            }
          }, 500);
        }, 200);
      };

      ws.onmessage = (e) => {
        try {
          let data: any;
          if (typeof e.data === 'string') {
            data = JSON.parse(e.data);
          } else {
            data = JSON.parse(new TextDecoder().decode(e.data as ArrayBuffer));
          }

          if (data.result) {
            addDebugLog(`Получено ${data.result.length} слов от Whisper`);
            data.result.forEach((word: any) => {
              results.push({
                word: word.word,
                start: word.start || 0,
                end: word.end || 0,
                confidence: word.confidence || word.conf || 0.85
              });
            });
          }

          if (data.text) {
            addDebugLog(`Whisper распознал текст: ${data.text.substring(0, 100)}...`);
            clearTimeout(timeoutId);
            ws.close();
            setProgress(100);
            resolve(results);
          }
        } catch (error) {
          console.error('Ошибка парсинга:', error);
        }
      };

      ws.onerror = (error) => {
        addDebugLog(`Ошибка WebSocket Whisper: ${error}`);
        clearTimeout(timeoutId);
        resolve(results);
      };

      ws.onclose = () => {
        addDebugLog('WebSocket Whisper закрыт');
        clearTimeout(timeoutId);
        resolve(results);
      };
    });
  };

  // Для Vosk оставляем чанки
  const processChunkWithRetry = async (
    chunk: any,
    chunkIndex: number,
    totalChunks: number,
    retryCount = 0
  ) => {
    return new Promise<any[]>((resolve) => {
      addDebugLog(`Чанк ${chunkIndex + 1}/${totalChunks} (${selectedModel.toUpperCase()}) (попытка ${retryCount + 1})`);
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
          if (data.result) {
            data.result.forEach((word: any) => {
              chunkResults.push({
                word: word.word,
                start: (word.start || 0) + chunk.startTime,
                end: (word.end || 0) + chunk.startTime,
                confidence: word.confidence || word.conf || 0.85,
                chunkIndex
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
      if (!m) return { start: 0, end: 0.35 };
      if (m.voskWord) return { start: m.voskWord.start, end: m.voskWord.end };
      const start = m.expectedStart ?? m.expectedTime ?? 0;
      const end = m.expectedEnd ?? Math.min(audioDuration, start + 0.35);
      return { start, end };
    };

    // Отмечаем слово как завершенное, если текущее время превысило его конец
    for (let i = 0; i < n; i++) {
      if (finalStateRef.current[i] !== 'pending') continue;
      const { end } = getSpan(i);
      if (currentTime > end + EPS) {
        finalStateRef.current[i] = 'finished';
      }
    }

    // Находим текущее слово (первое незавершенное)
    let currentWordIndex = -1;
    for (let i = 0; i < n; i++) {
      if (finalStateRef.current[i] === 'pending') {
        currentWordIndex = i;
        break;
      }
    }

    const newColors = new Array<string>(n);

    for (let i = 0; i < n; i++) {
      // Если слово уже завершено (произнесено) - берем цвет из iptColorsRef
      if (finalStateRef.current[i] !== 'pending') {
        newColors[i] = iptColorsRef.current[i] || COLORS_WORD.PENDING;
      }
      // Если это текущее слово
      else if (i === currentWordIndex) {
        newColors[i] = COLORS_WORD.CURRENT;
      }
      // Остальные - ожидающие
      else {
        newColors[i] = COLORS_WORD.PENDING;
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
    setCurrentPlayingWordIndex(currentWordIndex);
    setAudioCurrentTime(currentTime);
  };

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
      const resetColors = Array(referenceWordsRef.current.length).fill(COLORS_WORD.PENDING);
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
    isHighlightingRef.current = false;

    const n = referenceWordsRef.current.length;
    // Копируем цвета из iptColorsRef
    const newColors = [...iptColorsRef.current];

    setWordColors(newColors);
    wordColorsRef.current = newColors;
    finalStateRef.current = Array(n).fill('finished');

    addDebugLog('Аудио завершено, применены цвета ИПТ');
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

  const processFile = async (selectedFile: File) => {
    console.clear();
    addDebugLog(`Начало обработки файла (${selectedModel.toUpperCase()})`);

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
    setWordColors(Array(referenceWordsRef.current.length).fill(COLORS_WORD.PENDING));
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
    setAggregateIPT(null);
    wordMappingRef.current = [];
    isHighlightingRef.current = false;
    wordColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS_WORD.PENDING);
    finalStateRef.current = Array(referenceWordsRef.current.length).fill('pending');
    iptColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS_WORD.PENDING);
    audioUrlRef.current = URL.createObjectURL(selectedFile);
    try {
      setRecognitionStatus('Чтение и декодирование аудио...');
      const arrayBuffer = await readFileAsArrayBuffer(selectedFile);
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      let audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      addDebugLog('🎛️ Обработка шумов радиопереговоров...');
      audioBuffer = await enhanceAudioForVosk(audioBuffer);
      audioBufferRef.current = audioBuffer;
      const duration = audioBuffer.duration;
      addDebugLog(`Аудио: ${duration.toFixed(2)} секунд`);
      setAudioDuration(duration);

      let rawResults: any[] = [];

      if (selectedModel === 'whisper') {
        // Whisper - отправляем весь файл целиком
        setRecognitionStatus('Распознавание речи');
        setProgress(10);
        rawResults = await processFullAudioWhisper(audioBuffer);
      } else {
        // Vosk и Wav2Vec2 - разбиваем на чанки
        setRecognitionStatus('Разбиение аудио на чанки...');
        const chunks = splitAudioIntoChunks(audioBuffer, CHUNK_DURATION, 1.0);
        setRecognitionStatus(`Распознавание речи`);
        rawResults = await processAllChunks(chunks);
      }

      if (isCancelledRef.current) {
        addDebugLog('Обработка отменена');
        setIsProcessing(false);
        return;
      }
      const uniqueResults = deduplicateResults(rawResults);
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
    setFile(null);
    setIsProcessing(false);
    setRecognitionStatus('');
    setTranscription('');
    setPartial('');
    setWordColors(Array(referenceWordsRef.current.length).fill(COLORS_WORD.PENDING));
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
    setAggregateIPT(null);
    wordMappingRef.current = [];
    allResultsRef.current = [];
    audioBufferRef.current = null;
    wordColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS_WORD.PENDING);
    finalStateRef.current = Array(referenceWordsRef.current.length).fill('pending');
    iptColorsRef.current = Array(referenceWordsRef.current.length).fill(COLORS_WORD.PENDING);
    addDebugLog('Новая попытка');
  };

  const getAudioUrl = () => audioUrlRef.current || '';

  const scrollToWord = (startTime: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = startTime;
      audioRef.current.play();
    }
  };

  const ModelSelector = () => (
    <div style={{
      display: 'flex',
      gap: '0.75rem',
      background: COLORS_UI.bgCard,
      borderRadius: '20px',
      padding: '0.5rem',
      border: `1px solid ${COLORS_UI.border}`,
      boxShadow: `0 2px 4px ${COLORS_UI.shadow}`,
      marginBottom: '1rem'
    }}>
      {(['vosk', 'whisper', 'wav2vec2'] as const).map((model) => (
        <button
          key={model}
          onClick={() => !isProcessing && setSelectedModel(model)}
          disabled={isProcessing}
          style={{
            flex: 1,
            padding: '0.75rem 1rem',
            borderRadius: '16px',
            fontSize: '0.875rem',
            fontWeight: 600,
            border: 'none',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
            background: selectedModel === model ? COLORS_UI.primary : 'transparent',
            color: selectedModel === model ? COLORS_UI.textInverse : COLORS_UI.textSecondary,
            opacity: isProcessing ? 0.5 : 1,
            letterSpacing: '0.3px'
          }}
        >
          {model === 'vosk' ? 'Vosk' : model === 'whisper' ? 'Whisper' : 'Wav2Vec2'}
        </button>
      ))}
    </div>
  );

  return (
    <div style={{
      minHeight: '100vh',
      background: COLORS_UI.bgSurface,
      padding: 0,
      margin: 0
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '2rem' }}>

        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{
            fontSize: '2.25rem',
            fontWeight: '600',
            color: COLORS_UI.textPrimary,
            letterSpacing: '-0.02em',
            marginBottom: '0.5rem'
          }}>
            Речевой тренажер
          </h1>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: '1.5rem', marginBottom: '1.5rem', alignItems: 'start' }}>

          <div style={{ minWidth: 0 }}>
            {!file ? (
              <div style={{
                background: COLORS_UI.bgCard,
                borderRadius: '24px',
                padding: '1.5rem',
                border: `1px solid ${COLORS_UI.border}`,
                boxShadow: `0 4px 12px ${COLORS_UI.shadow}`,
                transition: 'all 0.2s ease'
              }}>
                <label style={{
                  display: 'block',
                  fontSize: '0.875rem',
                  fontWeight: '600',
                  color: COLORS_UI.textSecondary,
                  marginBottom: '0.875rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Эталонный текст
                </label>
                <textarea
                  value={referenceText}
                  onChange={(e) => updateReferenceText(e.target.value)}
                  disabled={isProcessing}
                  style={{
                    width: '100%',
                    padding: '1rem',
                    border: `2px solid ${COLORS_UI.border}`,
                    borderRadius: '20px',
                    fontSize: '1rem',
                    lineHeight: '1.6',
                    color: COLORS_UI.textPrimary,
                    background: isProcessing ? COLORS_UI.bgSurface : COLORS_UI.bgCard,
                    resize: 'vertical',
                    minHeight: '220px',
                    outline: 'none',
                    fontFamily: 'inherit',
                    boxSizing: 'border-box',
                    transition: 'border-color 0.2s ease'
                  }}
                  placeholder="Введите текст для анализа..."
                />
                <div style={{ fontSize: '0.875rem', color: COLORS_UI.textTertiary, marginTop: '0.875rem', fontWeight: '500' }}>
                  {referenceWordsRef.current.length} слов
                </div>
              </div>
            ) : (
              <div style={{
                background: COLORS_UI.bgCard,
                borderRadius: '24px',
                padding: '1.5rem',
                border: `1px solid ${COLORS_UI.border}`,
                boxShadow: `0 4px 12px ${COLORS_UI.shadow}`,
                transition: 'all 0.2s ease'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <div style={{ fontSize: '0.875rem', fontWeight: '600', color: COLORS_UI.textSecondary, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Эталонный текст
                  </div>
                  <div style={{ fontSize: '0.75rem', color: COLORS_UI.textTertiary, background: COLORS_UI.bgSurface, padding: '0.375rem 0.875rem', borderRadius: '16px', fontWeight: '500' }}>
                    {referenceWordsRef.current.length} слов
                  </div>
                </div>
                <div style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.75rem',
                  maxHeight: '340px',
                  overflowY: 'auto',
                  overflowX: 'hidden'
                }}>
                  {referenceWordsRef.current.map((word, index) => {
                    const color = wordColors[index] || COLORS_WORD.PENDING;
                    const styles: Record<string, React.CSSProperties> = {
                      [COLORS_WORD.CURRENT]: { background: `${COLORS_UI.current}12`, color: COLORS_UI.current, borderColor: `${COLORS_UI.current}40`, fontWeight: 600, fontSize: '1rem', padding: '0.5rem 1rem' },
                      [COLORS_WORD.CORRECT]: { background: `${COLORS_UI.success}12`, color: COLORS_UI.success, borderColor: `${COLORS_UI.success}30`, fontSize: '0.95rem', padding: '0.4rem 0.875rem' },
                      [COLORS_WORD.GOOD]: { background: `${COLORS_UI.primary}12`, color: COLORS_UI.primary, borderColor: `${COLORS_UI.primary}30`, fontSize: '0.95rem', padding: '0.4rem 0.875rem' },
                      [COLORS_WORD.ERROR]: { background: `${COLORS_UI.warning}12`, color: '#9a3412', borderColor: `${COLORS_UI.warning}30`, fontSize: '0.95rem', padding: '0.4rem 0.875rem' },
                      [COLORS_WORD.MISSED]: { background: `${COLORS_UI.error}12`, color: '#991b1b', borderColor: `${COLORS_UI.error}30`, fontSize: '0.95rem', padding: '0.4rem 0.875rem' },
                      [COLORS_WORD.PENDING]: { background: COLORS_UI.bgSurface, color: COLORS_UI.textTertiary, borderColor: COLORS_UI.border, fontSize: '0.95rem', padding: '0.4rem 0.875rem' }
                    };
                    return (
                      <span
                        key={index}
                        style={{
                          borderRadius: '14px',
                          fontWeight: color === COLORS_WORD.CURRENT ? '600' : '500',
                          border: '2px solid',
                          transition: 'all 0.2s ease',
                          whiteSpace: 'nowrap',
                          ...styles[color]
                        }}
                      >
                        {cleanWord(word)}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

            <ModelSelector />

            <div
              onClick={() => !isProcessing && document.getElementById('file-input')?.click()}
              style={{
                background: COLORS_UI.bgCard,
                borderRadius: '24px',
                padding: '1.25rem',
                border: `2px dashed ${isProcessing ? COLORS_UI.warning : file ? COLORS_UI.success : COLORS_UI.borderDark}`,
                textAlign: 'center',
                cursor: isProcessing ? 'default' : 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: `0 4px 12px ${COLORS_UI.shadow}`
              }}
            >
              <input id="file-input" type="file" accept="audio/*" onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])} disabled={isProcessing} style={{ display: 'none' }} />

              {isProcessing ? (
                <>
                  <div style={{ display: 'inline-block', marginBottom: '0.75rem' }}>
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ animation: 'spin 1s linear infinite' }}>
                      <circle cx="12" cy="12" r="10" stroke={COLORS_UI.warning} strokeWidth="2" strokeDasharray="30 30" strokeLinecap="round" fill="none" />
                    </svg>
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: '500', color: COLORS_UI.textPrimary }}>{recognitionStatus}</div>
                  <div style={{ marginTop: '0.75rem' }}>
                    <div style={{ height: '6px', background: COLORS_UI.border, borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ width: `${progress}%`, height: '100%', background: COLORS_UI.primary, transition: 'width 0.3s ease', borderRadius: '3px' }} />
                    </div>
                  </div>
                  {/* {totalSegments > 0 && (
                    <div style={{ fontSize: '0.75rem', color: COLORS_UI.textTertiary, marginTop: '0.5rem', fontWeight: '500' }}>
                      {currentSegment} / {totalSegments}
                    </div>
                  )} */}
                </>
              ) : file ? (
                <>
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={COLORS_UI.success} strokeWidth="1.5" style={{ marginBottom: '0.5rem' }}>
                    <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" stroke="currentColor" fill="none" />
                    <path d="M7 9l5-5 5 5" stroke="currentColor" fill="none" strokeLinecap="round" />
                    <path d="M12 4v12" stroke="currentColor" fill="none" strokeLinecap="round" />
                  </svg>
                  <div style={{ fontSize: '0.95rem', fontWeight: '500', color: COLORS_UI.textPrimary }}>{file.name}</div>
                  <div style={{ fontSize: '0.75rem', color: COLORS_UI.textTertiary, marginTop: '0.25rem' }}>{audioDuration.toFixed(1)} сек</div>
                </>
              ) : (
                <>
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={COLORS_UI.textTertiary} strokeWidth="1.5" style={{ marginBottom: '0.5rem' }}>
                    <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" stroke="currentColor" fill="none" />
                    <path d="M7 9l5-5 5 5" stroke="currentColor" fill="none" strokeLinecap="round" />
                    <path d="M12 4v12" stroke="currentColor" fill="none" strokeLinecap="round" />
                  </svg>
                  <div style={{ fontSize: '0.95rem', fontWeight: '500', color: COLORS_UI.textSecondary }}>Загрузить аудио</div>
                </>
              )}
            </div>

            {file && (
              <div style={{
                background: COLORS_UI.bgCard,
                borderRadius: '20px',
                padding: '0.75rem',
                border: `1px solid ${COLORS_UI.border}`,
                boxShadow: `0 4px 12px ${COLORS_UI.shadow}`,
                transition: 'all 0.2s ease'
              }}>
                <audio
                  ref={audioRef}
                  controls
                  src={getAudioUrl()}
                  style={{ width: '100%', height: '48px' }}
                  onPlay={handleAudioPlay}
                  onPause={handleAudioPause}
                  onEnded={handleAudioEnded}
                  onLoadedMetadata={handleAudioLoadedMetadata}
                />
              </div>
            )}
          </div>
        </div>

        {aggregateIPT && aggregateIPT.words.length > 0 && (
          <IPTChart
            words={aggregateIPT.words}
            currentTime={audioCurrentTime}
            onWordClick={scrollToWord}
            audioDuration={audioDuration}
            statsCounts={{
              perfect: aggregateIPT.perfectCount,
              good: aggregateIPT.goodCount,
              medium: aggregateIPT.mediumCount,
              poor: aggregateIPT.poorCount,
              critical: aggregateIPT.criticalCount
            }}
          />
        )}

        {(!isProcessing && aggregateIPT && aggregateIPT.words.length > 0) && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
            <WordIPTList
              words={aggregateIPT.words}
              currentTime={audioCurrentTime}
              onWordClick={scrollToWord}
              wordTimestamps={wordTimestamps}
            />
            <RecognizedTextBlock
              transcription={transcription}
              wordTimestamps={wordTimestamps}
              onWordClick={scrollToWord}
            />
          </div>
        )}

        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          audio::-webkit-media-controls-panel {
            background-color: ${COLORS_UI.bgSurface};
          }
        `}</style>
      </div>
    </div>
  );
}