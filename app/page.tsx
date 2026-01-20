'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

import type {
  ActivationFn,
  ExperimentResult,
  TrainingPoint,
  TrainingResult,
} from '@/lib/types';

import { getSalesDataset } from '@/lib/nn-sales';
import { getSatisfactionDataset } from '@/lib/nn-satisfaction';

type ModelKey = 'satisfaction' | 'sales';
type TabKey = 'resultados' | 'visualizacion' | 'explicacion';

type TrainApiResponse = {
  model: ModelKey;
  hyperparameters: {
    epochs: number;
    learningRate: number;
    activation: ActivationFn;
    seed?: number;
    logEvery: number;
  };
  result: TrainingResult;
  experiment?: ExperimentResult; // nuevo (modo laboratorio)
};

const activationForward = (z: number, fn: ActivationFn) => {
  switch (fn) {
    case 'tanh':
      return Math.tanh(z);
    case 'relu':
      return z > 0 ? z : 0;
    case 'sigmoid':
    default:
      return 1 / (1 + Math.exp(-z));
  }
};

/** Evita líneas gigantes si el peso crece mucho (mejor UX para el visual). */
const clampStrokeWidth = (w: number) => {
  const base = Math.abs(w) * 4;
  return Math.min(10, Math.max(1.5, base));
};

type Dataset = { X: [number, number][], y: number[] };

function getDataset(model: ModelKey): Dataset {
  return model === 'sales' ? getSalesDataset() : getSatisfactionDataset();
}

/** Mapea de “espacio de datos” a “espacio SVG” */
function makeScaler(domainMin: number, domainMax: number, rangeMin: number, rangeMax: number) {
  const denom = domainMax - domainMin || 1;
  return (v: number) => rangeMin + ((v - domainMin) / denom) * (rangeMax - rangeMin);
}

/** Descarga JSON (export experimento) */
function downloadJson(filename: string, obj: unknown) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function Home() {
  // === Estado principal (single run) ===
  const [data, setData] = useState<TrainingPoint[]>([]);
  const [displayedData, setDisplayedData] = useState<TrainingPoint[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [weights, setWeights] = useState<number[]>([]);
  const [bias, setBias] = useState<number>(0);

  // === Estado comparación A/B ===
  const [compareMode, setCompareMode] = useState(false);
  const [leftRun, setLeftRun] = useState<ExperimentResult | null>(null);
  const [rightRun, setRightRun] = useState<ExperimentResult | null>(null);

  const [model, setModel] = useState<ModelKey>('satisfaction');
  const [loading, setLoading] = useState(false);

  // Hiperparámetros base
  const [epochs, setEpochs] = useState(10_000);
  const [learningRate, setLearningRate] = useState(0.01);
  const [activation, setActivation] = useState<ActivationFn>('sigmoid');

  // Seed opcional (reproducibilidad)
  const [useSeed, setUseSeed] = useState(true);
  const [seed, setSeed] = useState(42);

  // Para comparación: LR/activación derecha (A/B)
  const [learningRateB, setLearningRateB] = useState(0.05);
  const [activationB, setActivationB] = useState<ActivationFn>('relu');

  // Inputs visibles para forward pass (didáctico)
  const [x1, setX1] = useState(1);
  const [x2, setX2] = useState(1);

  // Tabs + timeline
  const [activeTab, setActiveTab] = useState<TabKey>('resultados');
  const [stepIndex, setStepIndex] = useState(0);
  const [stepMode, setStepMode] = useState(false);

  // Timeline controls
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(80);

  // Pulso visual sincronizado con cambios de epoch/paso
  const [pulse, setPulse] = useState(false);

  // Metadatos del backend
  const [trainMeta, setTrainMeta] = useState<TrainApiResponse['hyperparameters'] | null>(null);

  // Último experimento para export/import
  const [lastExperiment, setLastExperiment] = useState<ExperimentResult | null>(null);

  // Intervalo para animación / playback (limpieza segura)
  const intervalRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const clearAnim = () => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => clearAnim();
  }, []);

  // Derivados
  const currentPoint = displayedData.at(-1);
  const currentError = currentPoint?.error ?? 0;
  const currentEpoch = currentPoint?.epoch ?? 0;

  // Detectar cambios → pulso
  useEffect(() => {
    if (currentEpoch > 0) {
      setPulse(true);
      const t = window.setTimeout(() => setPulse(false), 350);
      return () => window.clearTimeout(t);
    }
  }, [currentEpoch]);

  // Forward pass coherente con activación seleccionada (single run)
  const z = useMemo(() => {
    const w1 = weights[0] ?? 0;
    const w2 = weights[1] ?? 0;
    return x1 * w1 + x2 * w2 + (bias ?? 0);
  }, [x1, x2, weights, bias]);

  const yHat = useMemo(() => activationForward(z, activation), [z, activation]);

  // Dataset bounds para frontera de decisión
  const dataset = useMemo(() => getDataset(model), [model]);
  const bounds = useMemo(() => {
    const xs = dataset.X.map((p) => p[0]);
    const ys = dataset.X.map((p) => p[1]);
    return {
      xMin: Math.min(...xs),
      xMax: Math.max(...xs),
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
    };
  }, [dataset]);

  // IDs ARIA tabs
  const tabIds = {
    resultados: 'tab-resultados',
    visualizacion: 'tab-visualizacion',
    explicacion: 'tab-explicacion',
  } as const;

  const panelIds = {
    resultados: 'panel-resultados',
    visualizacion: 'panel-visualizacion',
    explicacion: 'panel-explicacion',
  } as const;

  // --- Fetch (single run) ---
  const fetchTrain = async (params: {
    model: ModelKey;
    epochs: number;
    learningRate: number;
    activation: ActivationFn;
    seed?: number;
  }): Promise<TrainApiResponse> => {
    const qs = new URLSearchParams({
      model: params.model,
      epochs: String(params.epochs),
      learningRate: String(params.learningRate),
      activation: params.activation,
    });
    if (params.seed !== undefined) qs.set('seed', String(params.seed));

    const res = await fetch(`/api/train?${qs.toString()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Train API failed: ${res.status}`);
    return (await res.json()) as TrainApiResponse;
  };

  // =========================
  // Timeline controls (single)
  // =========================
  const stopPlayback = () => {
    setPlaying(false);
    clearAnim();
  };

  const startPlayback = (history: TrainingPoint[], startAt: number) => {
    clearAnim();
    setPlaying(true);

    let index = startAt;
    intervalRef.current = window.setInterval(() => {
      if (index < history.length) {
        setDisplayedData(history.slice(0, index + 1));
        setStepIndex(index);
        index++;
      } else {
        stopPlayback();
      }
    }, speedMs);
  };

  useEffect(() => {
    // Si cambia speed mientras está reproduciendo, reiniciar con el mismo índice
    if (playing && data.length > 0) {
      startPlayback(data, stepIndex);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speedMs]);

  // =========================
  // Entrenamiento single run
  // =========================
  const handleTrainFull = async () => {
    stopPlayback();
    setLoading(true);

    try {
      const payload = await fetchTrain({
        model,
        epochs,
        learningRate,
        activation,
        seed: useSeed ? seed : undefined,
      });

      const result = payload.result;

      setTrainMeta(payload.hyperparameters);
      setData(result.history);
      setWeights(result.weights);
      setBias(result.bias);
      setPrediction(result.prediction);

      // Guardar experimento si viene (modo laboratorio)
      if (payload.experiment) setLastExperiment(payload.experiment);

      // Modo “auto”: reproducir timeline
      setDisplayedData([]);
      setStepIndex(0);
      startPlayback(result.history, 0);
    } catch (err) {
      console.error(err);
      setLoading(false);
    } finally {
      // En playback se apaga al terminar, pero aquí aseguramos no dejar loading pegado si falló.
      setLoading(false);
    }
  };

  const handleTrainStepMode = async () => {
    stopPlayback();
    setLoading(true);

    try {
      const payload = await fetchTrain({
        model,
        epochs,
        learningRate,
        activation,
        seed: useSeed ? seed : undefined,
      });

      const result = payload.result;

      setTrainMeta(payload.hyperparameters);
      setData(result.history);
      setDisplayedData(result.history.slice(0, 1));
      setStepIndex(0);

      setPrediction(result.prediction);
      setWeights(result.weights);
      setBias(result.bias);

      if (payload.experiment) setLastExperiment(payload.experiment);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleNextStep = () => {
    stopPlayback();
    if (stepIndex < data.length - 1) {
      const nextIndex = stepIndex + 1;
      setStepIndex(nextIndex);
      setDisplayedData(data.slice(0, nextIndex + 1));
    }
  };

  const handlePrevStep = () => {
    stopPlayback();
    if (stepIndex > 0) {
      const prev = stepIndex - 1;
      setStepIndex(prev);
      setDisplayedData(data.slice(0, prev + 1));
    }
  };

  const handleReset = () => {
    stopPlayback();
    setData([]);
    setDisplayedData([]);
    setStepIndex(0);
    setPrediction(null);
    setWeights([]);
    setBias(0);
    setTrainMeta(null);
    setLastExperiment(null);
    setLeftRun(null);
    setRightRun(null);
  };

  // =========================
  // Comparación A/B
  // =========================
  const handleTrainCompare = async () => {
    stopPlayback();
    setLoading(true);

    try {
      const baseSeed = useSeed ? seed : undefined;

      const [a, b] = await Promise.all([
        fetchTrain({
          model,
          epochs,
          learningRate,
          activation,
          seed: baseSeed,
        }),
        fetchTrain({
          model,
          epochs,
          learningRate: learningRateB,
          activation: activationB,
          seed: baseSeed,
        }),
      ]);

      // Preferimos el payload.experiment (modo laboratorio)
      setLeftRun(a.experiment ?? null);
      setRightRun(b.experiment ?? null);

      // También dejamos el “single run” mostrando A por defecto
      setTrainMeta(a.hyperparameters);
      setData(a.result.history);
      setDisplayedData(a.result.history.slice(0, 1));
      setWeights(a.result.weights);
      setBias(a.result.bias);
      setPrediction(a.result.prediction);
      setStepIndex(0);

      if (a.experiment) setLastExperiment(a.experiment);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // =========================
  // Export / Import
  // =========================
  const handleExport = () => {
    if (!lastExperiment) return;
    downloadJson(
      `rrnn-experimento-${lastExperiment.config.model}-${Date.now()}.json`,
      lastExperiment
    );
  };

  const handlePickImport = () => fileInputRef.current?.click();

  const handleImportFile: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as ExperimentResult;

      // Aplicar al estado (single run)
      setLastExperiment(parsed);
      setModel(parsed.config.model);
      setEpochs(parsed.config.epochs);
      setLearningRate(parsed.config.learningRate);
      setActivation(parsed.config.activation);
      setSeed(parsed.config.seed ?? 42);

      setTrainMeta({
        epochs: parsed.meta.epochs,
        learningRate: parsed.meta.learningRate,
        activation: parsed.meta.activation,
        seed: parsed.meta.seed,
        logEvery: parsed.meta.logEvery,
      });

      setData(parsed.result.history);
      setDisplayedData(parsed.result.history.slice(0, 1));
      setWeights(parsed.result.weights);
      setBias(parsed.result.bias);
      setPrediction(parsed.result.prediction);
      setStepIndex(0);
      stopPlayback();
    } catch (err) {
      console.error('Import failed:', err);
    } finally {
      // permitir re-import del mismo archivo
      e.target.value = '';
    }
  };

  // =========================
  // Frontera de decisión 2D
  // =========================
  const decisionLine = useMemo(() => {
    // w1*x + w2*y + b = 0  => y = (-b - w1*x)/w2
    const w1 = weights[0] ?? 0;
    const w2 = weights[1] ?? 0;
    const bb = bias ?? 0;

    // Si w2 es ~0, la línea es vertical: x = -b/w1
    if (Math.abs(w2) < 1e-6) {
      const x = Math.abs(w1) < 1e-6 ? null : -bb / w1;
      return { verticalX: x, p1: null as null | [number, number], p2: null as null | [number, number] };
    }

    const xA = bounds.xMin;
    const xB = bounds.xMax;
    const yA = (-bb - w1 * xA) / w2;
    const yB = (-bb - w1 * xB) / w2;
    return { verticalX: null as number | null, p1: [xA, yA] as [number, number], p2: [xB, yB] as [number, number] };
  }, [weights, bias, bounds]);

  // =========================
  // Explicación viva (simple)
  // =========================
  const liveMessage = useMemo(() => {
    const last = displayedData.at(-1);
    if (!last) return 'Entrena el modelo para ver qué ocurre paso a paso.';

    // Saturación (heurística): sigmoid cerca de 0/1 o tanh cerca de -1/1
    const yHatLocal = last.yHat;
    if (yHatLocal !== undefined) {
      if (activation === 'sigmoid' && (yHatLocal < 0.02 || yHatLocal > 0.98)) {
        return 'La neurona está saturada (ŷ muy cerca de 0 o 1). El gradiente tiende a hacerse pequeño.';
      }
      if (activation === 'tanh' && (yHatLocal < -0.98 || yHatLocal > 0.98)) {
        return 'La neurona está saturada (tanh cerca de -1 o 1). Cambios de pesos pueden volverse lentos.';
      }
      if (activation === 'relu' && yHatLocal === 0) {
        return 'ReLU está en 0 (neurona “apagada”). Si z permanece ≤ 0, el gradiente puede estancarse.';
      }
    }

    // Importancia relativa de pesos
    const w1 = weights[0] ?? 0;
    const w2 = weights[1] ?? 0;
    if (Math.abs(w1) > Math.abs(w2) * 1.5) return 'El modelo está usando más x₁ (|w₁| domina a |w₂|).';
    if (Math.abs(w2) > Math.abs(w1) * 1.5) return 'El modelo está usando más x₂ (|w₂| domina a |w₁|).';

    return 'Los pesos están relativamente balanceados: el modelo combina x₁ y x₂.';
  }, [displayedData, activation, weights]);

  return (
    <main className="h-screen flex flex-col bg-[#F8F7F6] overflow-hidden">
      {/* HEADER */}
      <header className="text-center py-4 border-b border-gray-200">
        <h1 className="text-3xl md:text-4xl font-serif text-black mb-1">
          Laboratorio de Aprendizaje Automático
        </h1>
        <p className="text-gray-600 text-sm md:text-base">
          Explora, ajusta y observa cómo aprende una red neuronal en tiempo real.
        </p>
      </header>

      {/* MAIN LAYOUT */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* PANEL DE CONTROL */}
        <aside
          className={`md:w-1/3 w-full bg-white/90 backdrop-blur-sm border-r border-gray-200 flex flex-col justify-between p-6 max-h-full transition-opacity duration-500 ${
            loading && !stepMode ? 'opacity-90' : 'opacity-100'
          }`}
        >
          <div className="overflow-y-auto flex-1 pr-2 space-y-4">
            {/* Selector de modelo */}
            <div>
              <label className="block text-xs uppercase text-black mb-1" htmlFor="model-select">
                Modelo
              </label>
              <select
                id="model-select"
                value={model}
                onChange={(e) => setModel(e.target.value as ModelKey)}
                className="w-full border rounded-lg px-3 py-2 focus:ring-[#A31F34] focus:outline-none text-black"
              >
                <option value="satisfaction">Satisfacción del Cliente</option>
                <option value="sales">Predicción de Ventas</option>
              </select>
            </div>

            {/* Activación */}
            <div>
              <label className="block text-xs uppercase text-black mb-1" htmlFor="activation-select">
                Activación (A)
              </label>
              <select
                id="activation-select"
                value={activation}
                onChange={(e) => setActivation(e.target.value as ActivationFn)}
                className="w-full border rounded-lg px-3 py-2 focus:ring-[#A31F34] focus:outline-none text-black"
              >
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
                <option value="relu">ReLU</option>
              </select>
            </div>

            {/* Compare mode toggle */}
            <div className="flex items-center gap-2 mt-2">
              <input
                id="compare-mode"
                type="checkbox"
                checked={compareMode}
                onChange={() => setCompareMode(!compareMode)}
                className="accent-[#A31F34] w-4 h-4"
              />
              <label htmlFor="compare-mode" className="text-sm text-gray-700">
                Modo comparación A/B
              </label>
            </div>

            {/* Config B solo si compareMode */}
            {compareMode && (
              <div className="rounded-lg border border-gray-200 p-3 bg-white">
                <p className="text-xs uppercase text-gray-500 mb-2">Configuración B</p>

                <label className="block text-xs text-gray-600 mb-1" htmlFor="activationB-select">
                  Activación (B)
                </label>
                <select
                  id="activationB-select"
                  value={activationB}
                  onChange={(e) => setActivationB(e.target.value as ActivationFn)}
                  className="w-full border rounded-lg px-3 py-2 focus:ring-[#A31F34] focus:outline-none text-black"
                >
                  <option value="sigmoid">Sigmoid</option>
                  <option value="tanh">Tanh</option>
                  <option value="relu">ReLU</option>
                </select>

                <label className="block text-xs text-gray-600 mb-1 mt-2" htmlFor="lrB-range">
                  Learning Rate (B)
                </label>
                <input
                  id="lrB-range"
                  type="range"
                  min={0.001}
                  max={0.2}
                  step={0.001}
                  value={learningRateB}
                  onChange={(e) => setLearningRateB(Number(e.target.value))}
                  className="w-full accent-[#A31F34]"
                />
                <p className="text-xs text-gray-600 mt-1">α(B) = {learningRateB.toFixed(3)}</p>
              </div>
            )}

            {/* Sliders */}
            <div>
              <label className="block text-xs uppercase text-gray-500 mb-1" htmlFor="epochs-range">
                Épocas
              </label>
              <input
                id="epochs-range"
                type="range"
                min={1000}
                max={20000}
                step={1000}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full accent-[#A31F34]"
              />
              <p className="text-xs text-gray-600 mt-1">{epochs} iteraciones</p>
            </div>

            <div>
              <label className="block text-xs uppercase text-gray-500 mb-1" htmlFor="lr-range">
                Learning Rate (A)
              </label>
              <input
                id="lr-range"
                type="range"
                min={0.001}
                max={0.2}
                step={0.001}
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full accent-[#A31F34]"
              />
              <p className="text-xs text-gray-600 mt-1">α(A) = {learningRate.toFixed(3)}</p>
            </div>

            {/* Seed */}
            <div className="rounded-lg border border-gray-200 p-3 bg-white">
              <div className="flex items-center gap-2">
                <input
                  id="use-seed"
                  type="checkbox"
                  checked={useSeed}
                  onChange={() => setUseSeed(!useSeed)}
                  className="accent-[#A31F34] w-4 h-4"
                />
                <label htmlFor="use-seed" className="text-sm text-gray-700">
                  Usar seed (reproducible)
                </label>
              </div>

              <div className="mt-2">
                <label className="block text-xs text-gray-600 mb-1" htmlFor="seed-input">
                  Seed
                </label>
                <input
                  id="seed-input"
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2 focus:ring-[#A31F34] focus:outline-none text-black"
                  disabled={!useSeed}
                />
              </div>
            </div>

            {/* Inputs (didáctico) */}
            <div className="pt-2">
              <p className="text-xs uppercase text-gray-500 mb-2">Inputs para la visualización</p>

              <label className="block text-xs text-gray-600 mb-1" htmlFor="x1-range">
                x₁ = {x1.toFixed(2)}
              </label>
              <input
                id="x1-range"
                type="range"
                min={bounds.xMin}
                max={bounds.xMax}
                step={(bounds.xMax - bounds.xMin) / 50 || 0.1}
                value={x1}
                onChange={(e) => setX1(Number(e.target.value))}
                className="w-full accent-[#A31F34]"
              />

              <label className="block text-xs text-gray-600 mb-1 mt-2" htmlFor="x2-range">
                x₂ = {x2.toFixed(2)}
              </label>
              <input
                id="x2-range"
                type="range"
                min={bounds.yMin}
                max={bounds.yMax}
                step={(bounds.yMax - bounds.yMin) / 50 || 0.5}
                value={x2}
                onChange={(e) => setX2(Number(e.target.value))}
                className="w-full accent-[#A31F34]"
              />
            </div>

            {/* Step Mode Toggle */}
            <div className="flex items-center gap-2 mt-4">
              <input
                id="step-mode"
                type="checkbox"
                checked={stepMode}
                onChange={() => {
                  setStepMode(!stepMode);
                  stopPlayback();
                }}
                className="accent-[#A31F34] w-4 h-4"
              />
              <label htmlFor="step-mode" className="text-sm text-gray-700">
                Modo paso a paso
              </label>
            </div>

            {/* Timeline controls */}
            <div className="rounded-lg border border-gray-200 p-3 bg-white">
              <p className="text-xs uppercase text-gray-500 mb-2">Timeline</p>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => (playing ? stopPlayback() : startPlayback(data, stepIndex))}
                  className="bg-gray-900 text-white px-3 py-1.5 rounded-md text-sm disabled:opacity-50"
                  disabled={data.length === 0}
                >
                  {playing ? '⏸ Pausa' : '▶ Play'}
                </button>

                <button
                  onClick={handlePrevStep}
                  className="bg-gray-200 text-black px-3 py-1.5 rounded-md text-sm disabled:opacity-50"
                  disabled={data.length === 0 || stepIndex === 0}
                >
                  ◀ Step
                </button>

                <button
                  onClick={handleNextStep}
                  className="bg-gray-200 text-black px-3 py-1.5 rounded-md text-sm disabled:opacity-50"
                  disabled={data.length === 0 || stepIndex >= data.length - 1}
                >
                  Step ▶
                </button>
              </div>

              <div className="mt-2">
                <label className="block text-xs text-gray-600 mb-1" htmlFor="speed-range">
                  Velocidad ({speedMs}ms)
                </label>
                <input
                  id="speed-range"
                  type="range"
                  min={30}
                  max={250}
                  step={10}
                  value={speedMs}
                  onChange={(e) => setSpeedMs(Number(e.target.value))}
                  className="w-full accent-[#A31F34]"
                  disabled={data.length === 0}
                />
              </div>
            </div>

            {/* Botones */}
            <div className="flex flex-col gap-2 mt-4">
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => {
                  if (compareMode) return handleTrainCompare();
                  return stepMode ? handleTrainStepMode() : handleTrainFull();
                }}
                disabled={loading}
                className="bg-gradient-to-r from-[#A31F34] to-[#8A1A2A] text-white py-2 rounded-md font-medium shadow-sm hover:shadow-md disabled:opacity-60 transition-all"
              >
                {loading
                  ? 'Entrenando...'
                  : compareMode
                  ? 'Entrenar A/B'
                  : stepMode
                  ? 'Iniciar Paso a Paso'
                  : 'Entrenar Modelo'}
              </motion.button>

              <button
                onClick={handleReset}
                className="bg-gray-200 text-black py-2 rounded-md font-medium hover:bg-gray-300 transition"
              >
                🔄 Reiniciar
              </button>

              {/* Export / Import */}
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={handleExport}
                  disabled={!lastExperiment}
                  className="bg-[#111827] text-white py-2 rounded-md font-medium disabled:opacity-50"
                >
                  ⬇ Export
                </button>
                <button
                  onClick={handlePickImport}
                  className="bg-[#FBBF24] text-black py-2 rounded-md font-medium hover:bg-[#FCD34D] transition"
                >
                  ⬆ Import
                </button>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/json"
                onChange={handleImportFile}
                className="hidden"
              />
            </div>
          </div>

          {/* Estado actual */}
          <div className="border-t border-gray-200 pt-4 mt-4 text-sm text-gray-600 space-y-1">
            <p>
              Época actual: <b>{currentEpoch}</b>
            </p>
            <p>
              Error actual: <b>{currentError.toFixed(4)}</b>
            </p>

            {trainMeta && (
              <p className="text-xs text-gray-500">
                act={trainMeta.activation} · logEvery={trainMeta.logEvery} · lr={trainMeta.learningRate} ·
                epochs={trainMeta.epochs}
              </p>
            )}
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <section className="flex-1 flex flex-col justify-center items-center p-6 relative overflow-hidden">
          {/* Tabs (A11y) */}
          <div
            role="tablist"
            aria-label="Secciones del laboratorio"
            className="flex justify-center gap-8 border-b border-gray-200 pb-2 mb-4"
          >
            {(['resultados', 'visualizacion', 'explicacion'] as const).map((tab) => (
              <button
                key={tab}
                id={tabIds[tab]}
                role="tab"
                aria-selected={activeTab === tab}
                aria-controls={panelIds[tab]}
                tabIndex={activeTab === tab ? 0 : -1}
                onClick={() => setActiveTab(tab)}
                className={`pb-1 text-sm font-medium transition ${
                  activeTab === tab
                    ? 'text-[#A31F34] border-b-2 border-[#A31F34]'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab === 'resultados'
                  ? '📊 Resultados'
                  : tab === 'visualizacion'
                  ? '🧠 Visualización'
                  : '🧮 Explicación'}
              </button>
            ))}
          </div>

          {/* Contenido dinámico */}
          <div className="w-full flex-1 flex justify-center items-center">
            {/* RESULTADOS */}
            {activeTab === 'resultados' && (
              <div
                id={panelIds.resultados}
                role="tabpanel"
                aria-labelledby={tabIds.resultados}
                className="relative w-full h-[85%] flex flex-col justify-center items-center gap-4"
              >
                {/* Comparación A/B */}
                {compareMode && leftRun && rightRun ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full h-full">
                    {[{ title: 'A', run: leftRun }, { title: 'B', run: rightRun }].map(({ title, run }) => (
                      <div key={title} className="bg-white rounded-xl border border-gray-200 p-3 h-full">
                        <p className="text-sm text-gray-700 mb-2">
                          <b>{title}</b> · act={run.meta.activation} · lr={run.meta.learningRate} · epochs={run.meta.epochs}
                        </p>
                        <div className="h-[320px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={run.result.history}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#E2E2E2" />
                              <XAxis dataKey="epoch" tick={{ fill: '#555' }} />
                              <YAxis tick={{ fill: '#555' }} />
                              <Tooltip
                                contentStyle={{
                                  backgroundColor: '#fff',
                                  borderRadius: '8px',
                                  border: '1px solid #A31F34',
                                }}
                              />
                              <Line
                                type="monotone"
                                dataKey="error"
                                stroke="#A31F34"
                                strokeWidth={2.4}
                                dot={false}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <>
                    {/* Single run chart */}
                    <div className="w-full h-[55%]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={displayedData.length > 0 ? displayedData : [{ epoch: 0, error: 0 }]}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E2E2E2" />
                          <XAxis dataKey="epoch" tick={{ fill: '#555' }} />
                          <YAxis tick={{ fill: '#555' }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#fff',
                              borderRadius: '8px',
                              border: '1px solid #A31F34',
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="error"
                            stroke="#A31F34"
                            strokeWidth={2.4}
                            dot={false}
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Decision boundary */}
                    <div className="w-full h-[45%] bg-white rounded-xl border border-gray-200 p-3">
                      <p className="text-sm text-gray-700 mb-2">
                        Frontera de decisión (w₁x₁ + w₂x₂ + b = 0)
                      </p>

                      <DecisionBoundary2D
                        dataset={dataset}
                        bounds={bounds}
                        weights={weights}
                        bias={bias}
                      />
                    </div>
                  </>
                )}
              </div>
            )}

            {/* VISUALIZACIÓN */}
            {activeTab === 'visualizacion' && (
              <div
                id={panelIds.visualizacion}
                role="tabpanel"
                aria-labelledby={tabIds.visualizacion}
                className="flex flex-col items-center gap-3"
              >
                <svg width="460" height="310" role="img" aria-label="Visualización de una neurona con dos entradas">
                  <title>Neurona: entradas, pesos, bias y salida</title>

                  {/* Inputs */}
                  <circle cx="55" cy="110" r="15" fill="#A31F34" />
                  <circle cx="55" cy="210" r="15" fill="#A31F34" />

                  {/* Conexiones */}
                  <motion.line
                    x1="70"
                    y1="110"
                    x2="230"
                    y2="160"
                    stroke={(weights[0] ?? 0) >= 0 ? '#16A34A' : '#DC2626'}
                    strokeWidth={
                      pulse ? clampStrokeWidth(weights[0] ?? 0) + 1.5 : clampStrokeWidth(weights[0] ?? 0)
                    }
                    transition={{ duration: 0.25, ease: 'easeOut' }}
                  />
                  <motion.line
                    x1="70"
                    y1="210"
                    x2="230"
                    y2="160"
                    stroke={(weights[1] ?? 0) >= 0 ? '#16A34A' : '#DC2626'}
                    strokeWidth={
                      pulse ? clampStrokeWidth(weights[1] ?? 0) + 1.5 : clampStrokeWidth(weights[1] ?? 0)
                    }
                    transition={{ duration: 0.25, ease: 'easeOut' }}
                  />

                  {/* Neurona */}
                  <motion.circle
                    cx="230"
                    cy="160"
                    r="30"
                    animate={{
                      fill: `rgba(163,31,52,${Math.min(1, Math.max(0.05, Math.abs(yHat)))})`,
                      scale: pulse ? 1.08 : 1,
                    }}
                    transition={{ duration: 0.35, ease: 'easeInOut' }}
                  />

                  {/* Output */}
                  <line x1="260" y1="160" x2="395" y2="160" stroke="#555" strokeWidth="2" />
                  <motion.circle
                    cx="395"
                    cy="160"
                    r="16"
                    animate={{
                      fill: `rgba(50,200,50,${Math.min(1, Math.max(0.05, Math.abs(prediction ?? 0.3)))})`,
                      scale: pulse ? 1.05 : 1,
                    }}
                    transition={{ duration: 0.25, ease: 'easeOut' }}
                  />

                  {/* Etiquetas */}
                  <text x="22" y="85" fontSize="12" fill="#444">
                    x₁
                  </text>
                  <text x="22" y="255" fontSize="12" fill="#444">
                    x₂
                  </text>
                  <text x="214" y="165" fontSize="12" fill="#fff">
                    {activation === 'sigmoid' ? 'σ' : activation === 'tanh' ? 'tanh' : 'ReLU'}
                  </text>
                </svg>

                <div className="text-xs text-gray-700 text-center max-w-xl">
                  <p>
                    z = x₁·w₁ + x₂·w₂ + b = <b>{z.toFixed(3)}</b> · ŷ = f(z) = <b>{yHat.toFixed(3)}</b>
                  </p>
                  <p className="mt-1">
                    <span className="inline-block w-3 h-3 bg-[#16A34A] align-middle mr-1 rounded-sm" />
                    peso positivo ·
                    <span className="inline-block w-3 h-3 bg-[#DC2626] align-middle mx-1 rounded-sm" />
                    peso negativo
                  </p>
                </div>
              </div>
            )}

            {/* EXPLICACIÓN */}
            {activeTab === 'explicacion' && (
              <div
                id={panelIds.explicacion}
                role="tabpanel"
                aria-labelledby={tabIds.explicacion}
                className="max-w-2xl text-center"
              >
                <h3 className="text-2xl font-serif text-black mb-3">🧮 Cómo Aprende una Red Neuronal</h3>

                <div className="bg-white border border-gray-200 rounded-xl p-4 text-left text-sm text-gray-700 mb-4">
                  <p className="font-semibold text-gray-900 mb-1">Explicación viva</p>
                  <p>{liveMessage}</p>
                </div>

                <motion.pre
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="bg-[#1B1B1B] text-[#F1F1F1] rounded-xl p-6 mb-6 font-mono text-lg md:text-xl border-l-4 border-[#D4AF37] shadow-[inset_0_0_20px_rgba(0,0,0,0.5)] text-left"
                >
{`Forward:
z = x₁w₁ + x₂w₂ + b
ŷ = f(z)

Loss (demo):
error = (y - ŷ)²

Update (idea):
wᵢ ← wᵢ + α * (y - ŷ) * f'(z) * xᵢ`}
                </motion.pre>

                <p className="text-gray-600 text-sm md:text-base">
                  En cada <strong>epoch</strong>, el modelo ajusta los pesos <em>(w₁, w₂)</em> y el sesgo{' '}
                  <em>(b)</em> para reducir el error. Con “Comparación A/B” puedes ver cómo cambian los resultados al
                  variar <em>learning rate</em> o la función de activación.
                </p>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* FOOTER */}
      <footer className="sticky bottom-0 bg-[#1B1B1B] text-gray-300 py-3 px-6 flex justify-between items-center text-sm">
        <p>© 2025 Laboratorio de Aprendizaje Automático · Ing. Emilio Bogantes</p>
        <p className="text-gray-500">{compareMode ? 'Comparación A/B' : stepMode ? 'Paso a Paso' : 'Auto'} · v6.0</p>
      </footer>
    </main>
  );
}

/**
 * Componente: Frontera de decisión 2D
 * - Dibuja puntos del dataset (y=0 rojo, y=1 verde)
 * - Dibuja línea w1*x + w2*y + b = 0
 *
 * Sin dependencias extra: SVG puro.
 */
function DecisionBoundary2D(props: {
  dataset: { X: [number, number][], y: number[] };
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
  weights: number[];
  bias: number;
}) {
  const { dataset, bounds, weights, bias } = props;

  const W = 520;
  const H = 220;
  const pad = 18;

  const sx = makeScaler(bounds.xMin, bounds.xMax, pad, W - pad);
  // Ojo: SVG y crece hacia abajo; invertimos para que “arriba” sea mayor y
  const sy = makeScaler(bounds.yMin, bounds.yMax, H - pad, pad);

  const w1 = weights[0] ?? 0;
  const w2 = weights[1] ?? 0;

  // Compute line in data space
  let line: { x1: number; y1: number; x2: number; y2: number } | null = null;
  if (Math.abs(w2) < 1e-6) {
    // vertical: x = -b/w1
    if (Math.abs(w1) >= 1e-6) {
      const x = -bias / w1;
      line = { x1: x, y1: bounds.yMin, x2: x, y2: bounds.yMax };
    }
  } else {
    const xA = bounds.xMin;
    const xB = bounds.xMax;
    const yA = (-bias - w1 * xA) / w2;
    const yB = (-bias - w1 * xB) / w2;
    line = { x1: xA, y1: yA, x2: xB, y2: yB };
  }

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Frontera de decisión en 2D">
      <title>Frontera de decisión y puntos del dataset</title>

      {/* Axes */}
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#999" strokeWidth="1" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#999" strokeWidth="1" />

      {/* Points */}
      {dataset.X.map(([x, yVal], i) => {
        const label = dataset.y[i];
        return (
          <circle
            key={i}
            cx={sx(x)}
            cy={sy(yVal)}
            r={5}
            fill={label === 1 ? '#16A34A' : '#DC2626'}
            opacity={0.9}
          />
        );
      })}

      {/* Decision boundary */}
      {line && (
        <line
          x1={sx(line.x1)}
          y1={sy(line.y1)}
          x2={sx(line.x2)}
          y2={sy(line.y2)}
          stroke="#111827"
          strokeWidth="2"
        />
      )}

      {/* Labels */}
      <text x={W - pad - 10} y={H - pad - 6} fontSize="11" fill="#555">
        x₁
      </text>
      <text x={pad + 6} y={pad + 12} fontSize="11" fill="#555">
        x₂
      </text>
    </svg>
  );
}