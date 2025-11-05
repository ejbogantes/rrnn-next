/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import { useState, useEffect } from 'react';
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

interface TrainingPoint {
  epoch: number;
  error: number;
}
interface TrainingResult {
  weights: number[];
  bias: number;
  prediction: number;
  history: TrainingPoint[];
}

export default function Home() {
  const [data, setData] = useState<TrainingPoint[]>([]);
  const [displayedData, setDisplayedData] = useState<TrainingPoint[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [weights, setWeights] = useState<number[]>([]);
  const [bias, setBias] = useState<number>(0);
  const [model, setModel] = useState<'satisfaction' | 'sales'>('satisfaction');
  const [loading, setLoading] = useState(false);
  const [epochs, setEpochs] = useState(10000);
  const [learningRate, setLearningRate] = useState(0.01);
  const [activeTab, setActiveTab] = useState<'resultados' | 'visualizacion' | 'explicacion'>('resultados');
  const [stepIndex, setStepIndex] = useState(0);
  const [stepMode, setStepMode] = useState(false);
  const [pulse, setPulse] = useState(false); // 🧠 Sincronización visual

  const currentError = displayedData.at(-1)?.error ?? 0;
  const currentEpoch = displayedData.at(-1)?.epoch ?? 0;

  // Detectar cambios en epoch → activar pulso
  useEffect(() => {
    if (currentEpoch > 0) {
      setPulse(true);
      const timeout = setTimeout(() => setPulse(false), 400);
      return () => clearTimeout(timeout);
    }
  }, [currentEpoch]);

  // === ENTRENAMIENTO AUTOMÁTICO ===
  const handleTrainFull = async () => {
    setLoading(true);
    const res = await fetch(`/api/train?model=${model}&epochs=${epochs}&learningRate=${learningRate}`);
    const result: TrainingResult = await res.json();
    setData(result.history);
    setWeights(result.weights);
    setBias(result.bias);
    setPrediction(result.prediction);

    // Animación progresiva del gráfico
    let index = 0;
    const interval = setInterval(() => {
      if (index < result.history.length) {
        setDisplayedData(result.history.slice(0, index + 1));
        index++;
      } else {
        clearInterval(interval);
        setLoading(false);
      }
    }, 60);
  };

  // === ENTRENAMIENTO PASO A PASO ===
  const handleTrainStepMode = async () => {
    setLoading(true);
    const res = await fetch(`/api/train?model=${model}&epochs=${epochs}&learningRate=${learningRate}`);
    const result: TrainingResult = await res.json();
    setData(result.history);
    setDisplayedData(result.history.slice(0, 1));
    setStepIndex(0);
    setPrediction(result.prediction);
    setWeights(result.weights);
    setBias(result.bias);
    setLoading(false);
  };

  const handleNextStep = () => {
    if (stepIndex < data.length - 1) {
      const nextIndex = stepIndex + 1;
      setStepIndex(nextIndex);
      setDisplayedData(data.slice(0, nextIndex + 1));
    }
  };

  const handleReset = () => {
    setData([]);
    setDisplayedData([]);
    setStepIndex(0);
    setPrediction(null);
    setWeights([]);
    setBias(0);
  };

  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

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
              <label className="block text-xs uppercase text-black mb-1">Modelo</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value as 'satisfaction' | 'sales')}
                className="w-full border rounded-lg px-3 py-2 focus:ring-[#A31F34] focus:outline-none text-black"
              >
                <option value="satisfaction">Satisfacción del Cliente</option>
                <option value="sales">Predicción de Ventas</option>
              </select>
            </div>

            {/* Sliders */}
            <div>
              <label className="block text-xs uppercase text-gray-500 mb-1">Épocas</label>
              <input
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
              <label className="block text-xs uppercase text-gray-500 mb-1">Learning Rate</label>
              <input
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

            {/* Step Mode Toggle */}
            <div className="flex items-center gap-2 mt-4">
              <input
                type="checkbox"
                checked={stepMode}
                onChange={() => setStepMode(!stepMode)}
                className="accent-[#A31F34] w-4 h-4"
              />
              <span className="text-sm text-gray-700">Modo paso a paso</span>
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
                {loading
                  ? 'Entrenando...'
                  : stepMode
                  ? 'Iniciar Paso a Paso'
                  : 'Entrenar Modelo'}
              </motion.button>

              {stepMode && data.length > 0 && (
                <>
                  <button
                    onClick={handleNextStep}
                    className="bg-[#FBBF24] text-black py-2 rounded-md font-medium hover:bg-[#FCD34D] transition"
                  >
                    ▶ Siguiente Epoch
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
          <div className="border-t border-gray-200 pt-4 mt-4 text-sm text-gray-600">
            <p>Época actual: <b>{currentEpoch}</b></p>
            <p>Error actual: <b>{currentError.toFixed(4)}</b></p>
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <section className="flex-1 flex flex-col justify-center items-center p-6 relative overflow-hidden">
          {/* Tabs */}
          <div className="flex justify-center gap-8 border-b border-gray-200 pb-2 mb-4">
            {['resultados', 'visualizacion', 'explicacion'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab as any)}
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
              <div className="relative w-full h-[80%] flex flex-col justify-center items-center">
                <ResponsiveContainer width="95%" height="100%">
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
              <svg width="440" height="270">
                {/* Inputs */}
                <circle cx="50" cy="100" r="15" fill="#A31F34" />
                <circle cx="50" cy="180" r="15" fill="#A31F34" />

                {/* Conexiones sincronizadas */}
                <motion.line
                  x1="65" y1="100" x2="210" y2="140"
                  stroke={weights[0] > 0 ? '#16A34A' : '#DC2626'}
                  strokeWidth={pulse ? Math.abs(weights[0] || 0.5) * 8 : Math.abs(weights[0] || 0.5) * 6}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
                <motion.line
                  x1="65" y1="180" x2="210" y2="140"
                  stroke={weights[1] > 0 ? '#16A34A' : '#DC2626'}
                  strokeWidth={pulse ? Math.abs(weights[1] || 0.5) * 8 : Math.abs(weights[1] || 0.5) * 6}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />

                {/* Neurona */}
                <motion.circle
                  cx="210" cy="140" r="26"
                  animate={{
                    fill: `rgba(163,31,52,${sigmoid((weights[0] || 0) + (weights[1] || 0))})`,
                    scale: pulse ? 1.1 : 1,
                  }}
                  transition={{ duration: 0.4, ease: 'easeInOut' }}
                />

                {/* Output */}
                <line x1="236" y1="140" x2="370" y2="140" stroke="#555" strokeWidth="2" />
                <motion.circle
                  cx="370" cy="140" r="16"
                  animate={{ fill: `rgba(50,200,50,${prediction || 0.3})`, scale: pulse ? 1.05 : 1 }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
              </svg>
            )}

            {/* EXPLICACIÓN */}
            {activeTab === 'explicacion' && (
              <div className="max-w-2xl text-gray-200 text-center">
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
                  En cada <strong>epoch</strong>, el modelo ajusta los pesos <em>(w₁, w₂)</em> y el sesgo <em>(b)</em>  
                  para minimizar el error y aprender el patrón subyacente.
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