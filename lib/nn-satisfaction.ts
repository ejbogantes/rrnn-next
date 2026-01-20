// lib/nn-satisfaction.ts

import { TrainingResult, TrainingPoint } from './types';

/**
 * PRNG local (determinista con seed).
 * Importante: NO sobrescribe Math.random (evita side-effects globales).
 */
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Entrena un modelo de red neuronal muy simple (una sola neurona)
 * para clasificar niveles de satisfacción.
 *
 * Este ejercicio es educativo: muestra forward pass, error, gradiente
 * y actualización de pesos/bias con gradiente descendente.
 */
export function trainSatisfaction(options?: {
  learningRate?: number; // Tasa de aprendizaje
  epochs?: number; // Número de épocas
  logEvery?: number; // Cada cuántas épocas guardar un punto en history
  seed?: number; // Semilla opcional para reproducibilidad
}): TrainingResult {
  const {
    learningRate: learningRateRaw = 0.01,
    epochs: epochsRaw = 2_000, // default amigable para UI/visualizaciones
    logEvery: logEveryRaw = 100,
    seed,
  } = options || {};

  // Validaciones suaves (evita NaN / negativos)
  const learningRate =
    Number.isFinite(learningRateRaw) && learningRateRaw > 0 ? learningRateRaw : 0.01;

  const epochs = Number.isFinite(epochsRaw) ? Math.max(1, Math.floor(epochsRaw)) : 2_000;

  const logEvery = Number.isFinite(logEveryRaw)
    ? Math.max(1, Math.floor(logEveryRaw))
    : 100;

  // Determinismo opcional, sin tocar Math.random global
  const rand = seed !== undefined ? mulberry32(seed) : Math.random;

  // Funciones auxiliares
  const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
  const sigmoidDerivative = (yHat: number): number => yHat * (1 - yHat);

  // Datos de entrenamiento
  // Nota didáctica: el comentario original decía "normalizados", pero estos valores NO están normalizados.
  // Puedes convertir esto en lección: escalas distintas afectan el entrenamiento y la magnitud de los pesos.
  const X: number[][] = [
    [2, 5],
    [10, 4],
    [25, 2],
    [30, 1],
    [5, 4],
  ];

  const y: number[] = [1, 1, 0, 0, 1];

  // Inicialización de parámetros (con rand local)
  const w = [rand(), rand()];
  let b = rand();

  const history: TrainingPoint[] = [];

  // Entrenamiento
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalError = 0;

    // Snapshot representativo de la época (último sample)
    let lastZ = 0;
    let lastYHat = 0;

    for (let i = 0; i < X.length; i++) {
      const [x1, x2] = X[i];

      // Forward pass
      const z = x1 * w[0] + x2 * w[1] + b;
      const yHat = sigmoid(z);

      lastZ = z;
      lastYHat = yHat;

      // Error
      const error = y[i] - yHat;

      // Gradiente (demo simple: MSE + sigmoide)
      // En clasificación real suele usarse cross-entropy, pero MSE funciona para visualización educativa.
      const gradient = error * sigmoidDerivative(yHat);

      // Update
      w[0] += learningRate * gradient * x1;
      w[1] += learningRate * gradient * x2;
      b += learningRate * gradient;

      totalError += Math.pow(error, 2);
    }

    // Log (para graficar y animar)
    if (epoch % logEvery === 0) {
      history.push({
        epoch,
        error: totalError / X.length,
        weights: [w[0], w[1]], // snapshot defensivo
        bias: b,
        z: lastZ,
        yHat: lastYHat,
      });
    }
  }

  // Evaluación final
  const testInput: [number, number] = [12, 3];
  const pred = sigmoid(testInput[0] * w[0] + testInput[1] * w[1] + b);

  return {
    weights: [w[0], w[1]], // snapshot defensivo (evita mutaciones accidentales)
    bias: b,
    prediction: pred,
    history,
  };
}