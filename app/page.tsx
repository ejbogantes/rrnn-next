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

import type { TrainingPoint, TrainingResult } from '@/lib/types';

type ModelKey = 'satisfaction' | 'sales';
type TabKey = 'resultados' | 'visualizacion' | 'explicacion';

type TrainApiResponse = {
  model: ModelKey;
  hyperparameters: {
    epochs: number;
    learningRate: number;
    seed?: number;
    logEvery: number;
  };
  result: TrainingResult;
};

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Evita líneas gigantes si el peso crece mucho (mejor UX para el visual). */
const clampStrokeWidth = (w: number) => {
  const base = Math.abs(w) * 4; // escala visual
  return Math.min(10, Math.max(1.5, base));
};

export default function Home() {
  const [data, setData] = useState<TrainingPoint[]>([]);
  const [displayedData, setDisplayedData] = useState<TrainingPoint[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [weights, setWeights] = useState<number[]>([]);
  const [bias, setBias] = useState<number>(0);

  const [model, setModel] = useState<ModelKey>('satisfaction');
  const [loading, setLoading] = useState(false);

  // Hiperparámetros (UI)
  const [epochs, setEpochs] = useState(10_000);
  const [learningRate, setLearningRate] = useState(0.01);

  // Inputs visibles para la visualización del forward pass (didáctico)
  const [x1, setX1] = useState(1);
  const [x2, setX2] = useState(1);

  // Tabs + step mode
  const [activeTab, setActiveTab] = useState<TabKey>('resultados');
  const [stepIndex, setStepIndex] = useState(0);
  const [stepMode, setStepMode] = useState(false);

  // Pulso visual sincronizado con cambios de epoch/paso
  const [pulse, setPulse] = useState(false);

  // Metadatos del backend (sirven para mostrar contexto del entrenamiento)
  const [trainMeta, setTrainMeta] = useState<TrainApiResponse['hyperparameters'] | null>(null);

  // Intervalo para animación del gráfico (limpieza segura)
  const intervalRef = useRef<number | null>(null);

  const clearAnim = () => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    // Cleanup al desmontar
    return () => clearAnim();
  }, []);

  const currentPoint = displayedData.at(-1);
  const currentError = currentPoint?.error ?? 0;
  const currentEpoch = currentPoint?.epoch ?? 0;

  // Detectar cambios en epoch → activar pulso
  useEffect(() => {
    if (currentEpoch > 0) {
      setPulse(true);
      const timeout = window.setTimeout(() => setPulse(false), 400);
      return () => window.clearTimeout(timeout);
    }
  }, [currentEpoch]);

  // Forward pass coherente con la explicación: z = x1*w1 + x2*w2 + b
  const z = useMemo(() => {
    const w1 = weights[0] ?? 0;
    const w2 = weights[1] ?? 0;
    return x1 * w1 + x2 * w2 + (bias ?? 0);
  }, [x1, x2, weights, bias]);

  const yHat = useMemo(() => sigmoid(z), [z]);

  // IDs ARIA para tabs
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

  const fetchTrain = async (): Promise<TrainApiResponse> => {
    const url = `/api/train?model=${encodeURIComponent(model)}&epochs=${encodeURIComponent(
      epochs
    )}&learningRate=${encodeURIComponent(learningRate)}`;

    const res = await fetch(url);
    if (!res.ok) {
      // Evita estados “silenciosos” en UI si el backend falla
      throw new Error(`Train API failed: ${res.status}`);
    }
    return (await res.json()) as TrainApiResponse;
  };

  // === ENTRENAMIENTO AUTOMÁTICO ===
  const handleTrainFull = async () => {
    clearAnim();
    setLoading(true);

    try {
      const payload = await fetchTrain();
      const result = payload.result;

      setTrainMeta(payload.hyperparameters);
      setData(result.history);
      setWeights(result.weights);
      setBias(result.bias);
      setPrediction(result.prediction);

      // Animación progresiva del gráfico (sin bloquear UI)
      let index = 0;
      intervalRef.current = window.setInterval(() => {
        if (index < result.history.length) {
          setDisplayedData(result.history.slice(0, index + 1));
          index++;
        } else {
          clearAnim();
          setLoading(false);
        }
      }, 60);
    } catch (err) {
      console.error(err);
      // Fallback seguro: detiene loading y deja UI usable
      setLoading(false);
    }
  };

  // === ENTRENAMIENTO PASO A PASO ===
  const handleTrainStepMode = async () => {
    clearAnim();
    setLoading(true);

    try {
      const payload = await fetchTrain();
      const result = payload.result;

      setTrainMeta(payload.hyperparameters);
      setData(result.history);
      setDisplayedData(result.history.slice(0, 1));
      setStepIndex(0);

      setPrediction(result.prediction);
      setWeights(result.weights);
      setBias(result.bias);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleNextStep = () => {
    if (stepIndex < data.length - 1) {
      const nextIndex = stepIndex + 1;
      setStepIndex(nextIndex);
      setDisplayedData(data.slice(0, nextIndex + 1));
    }
  };

  const handleReset = () => {
    clearAnim();
    setData([]);
    setDisplayedData([]);
    setStepIndex(0);
    setPrediction(null);
    setWeights([]);
    setBias(0);
    setTrainMeta(null);
  };

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
                Learning Rate
              </label>
              <input
                id="lr-range"
                type="range"
                min={0.001}
                max={0.05}
                step={0.001}
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full accent-[#A31F34]"
              />
              <p className="text-xs text-gray-600 mt-1">α = {learningRate.toFixed(3)}</p>
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
                min={0}
                max={5}
                step={0.1}
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
                min={0}
                max={20}
                step={0.5}
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
                onChange={() => setStepMode(!stepMode)}
                className="accent-[#A31F34] w-4 h-4"
              />
              <label htmlFor="step-mode" className="text-sm text-gray-700">
                Modo paso a paso
              </label>
            </div>

            {/* Botones */}
            <div className="flex flex-col gap-2 mt-4">
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={stepMode ? handleTrainStepMode : handleTrainFull}
                disabled={loading}
                className="bg-gradient-to-r from-[#A31F34] to-[#8A1A2A] text-white py-2 rounded-md font-medium shadow-sm hover:shadow-md disabled:opacity-60 transition-all"
              >
                {loading ? 'Entrenando...' : stepMode ? 'Iniciar Paso a Paso' : 'Entrenar Modelo'}
              </motion.button>

              {stepMode && data.length > 0 && (
                <>
                  <button
                    onClick={handleNextStep}
                    className="bg-[#FBBF24] text-black py-2 rounded-md font-medium hover:bg-[#FCD34D] transition"
                  >
                    ▶ Siguiente paso
                  </button>
                  <button
                    onClick={handleReset}
                    className="bg-gray-200 text-black py-2 rounded-md font-medium hover:bg-gray-300 transition"
                  >
                    🔄 Reiniciar
                  </button>
                </>
              )}
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
                logEvery={trainMeta.logEvery} · lr={trainMeta.learningRate} · epochs={trainMeta.epochs}
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
                className="relative w-full h-[80%] flex flex-col justify-center items-center"
              >
                <ResponsiveContainer width="95%" height="100%">
                  <LineChart
                    data={displayedData.length > 0 ? displayedData : [{ epoch: 0, error: 0 }]}
                  >
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
                      isAnimationActive
                      animationDuration={600}
                      animationEasing="ease-in-out"
                    />
                  </LineChart>
                </ResponsiveContainer>
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
                <svg width="440" height="300" role="img" aria-label="Visualización de una neurona con dos entradas">
                  <title>Neurona: entradas, pesos, bias y salida</title>

                  {/* Inputs */}
                  <circle cx="50" cy="110" r="15" fill="#A31F34" />
                  <circle cx="50" cy="200" r="15" fill="#A31F34" />

                  {/* Conexiones */}
                  <motion.line
                    x1="65"
                    y1="110"
                    x2="210"
                    y2="155"
                    stroke={(weights[0] ?? 0) >= 0 ? '#16A34A' : '#DC2626'}
                    strokeWidth={pulse ? clampStrokeWidth(weights[0] ?? 0) + 1.5 : clampStrokeWidth(weights[0] ?? 0)}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                  />
                  <motion.line
                    x1="65"
                    y1="200"
                    x2="210"
                    y2="155"
                    stroke={(weights[1] ?? 0) >= 0 ? '#16A34A' : '#DC2626'}
                    strokeWidth={pulse ? clampStrokeWidth(weights[1] ?? 0) + 1.5 : clampStrokeWidth(weights[1] ?? 0)}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                  />

                  {/* Neurona */}
                  <motion.circle
                    cx="210"
                    cy="155"
                    r="28"
                    animate={{
                      // La intensidad refleja yHat (activación) real del forward pass
                      fill: `rgba(163,31,52,${Math.min(1, Math.max(0.05, yHat))})`,
                      scale: pulse ? 1.08 : 1,
                    }}
                    transition={{ duration: 0.4, ease: 'easeInOut' }}
                  />

                  {/* Output */}
                  <line x1="238" y1="155" x2="370" y2="155" stroke="#555" strokeWidth="2" />
                  <motion.circle
                    cx="370"
                    cy="155"
                    r="16"
                    animate={{
                      fill: `rgba(50,200,50,${Math.min(1, Math.max(0.05, prediction ?? 0.3))})`,
                      scale: pulse ? 1.05 : 1,
                    }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                  />

                  {/* Etiquetas rápidas */}
                  <text x="20" y="85" fontSize="12" fill="#444">
                    x₁
                  </text>
                  <text x="20" y="245" fontSize="12" fill="#444">
                    x₂
                  </text>
                  <text x="195" y="160" fontSize="12" fill="#fff">
                    σ
                  </text>
                </svg>

                {/* Mini-leyenda + valores (didáctico, sin “magia”) */}
                <div className="text-xs text-gray-600 text-center">
                  <p>
                    z = x₁·w₁ + x₂·w₂ + b = <b>{z.toFixed(3)}</b> · ŷ = σ(z) = <b>{yHat.toFixed(3)}</b>
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
                className="max-w-2xl text-gray-200 text-center"
              >
                <h3 className="text-2xl font-serif text-black mb-5">🧮 Cómo Aprende una Red Neuronal</h3>
                <motion.pre
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6 }}
                  className="bg-[#1B1B1B] text-[#F1F1F1] rounded-xl p-6 mb-6 font-mono text-lg md:text-xl border-l-4 border-[#D4AF37] shadow-[inset_0_0_20px_rgba(0,0,0,0.5)]"
                >
{`ŷ = 1 / (1 + e^(-z))
z = (x₁w₁ + x₂w₂ + b)

error = (y - ŷ)²

wᵢ ← wᵢ + α * (y - ŷ) * ŷ * (1 - ŷ) * xᵢ`}
                </motion.pre>
                <p className="text-gray-600 text-sm md:text-base">
                  En cada <strong>epoch</strong>, el modelo ajusta los pesos <em>(w₁, w₂)</em> y el sesgo{' '}
                  <em>(b)</em> para minimizar el error y aprender el patrón subyacente.
                </p>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* FOOTER */}
      <footer className="sticky bottom-0 bg-[#1B1B1B] text-gray-300 py-3 px-6 flex justify-between items-center text-sm">
        <p>© 2025 Laboratorio de Aprendizaje Automático · Ing. Emilio Bogantes</p>
        <p className="text-gray-500">{stepMode ? 'Modo Paso a Paso' : 'Modo Automático'} · v5.3</p>
      </footer>
    </main>
  );
}