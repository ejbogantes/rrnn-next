// lib/nn-satisfaction.ts

import type { ActivationFn, TrainingPoint, TrainingResult } from './types';

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
 * Dataset del ejercicio de satisfacción.
 * Lo exportamos para poder:
 * - dibujar los puntos en 2D
 * - mostrar frontera de decisión
 * - explicar separabilidad / escalas
 */
export function getSatisfactionDataset(): { X: [number, number][], y: number[] } {
  const X: [number, number][] = [
    [2, 5],
    [10, 4],
    [25, 2],
    [30, 1],
    [5, 4],
  ];

  const y: number[] = [1, 1, 0, 0, 1];
  return { X, y };
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
function tanh(x: number): number {
  return Math.tanh(x);
}
function relu(x: number): number {
  return x > 0 ? x : 0;
}

function activationForward(z: number, fn: ActivationFn): number {
  switch (fn) {
    case 'tanh':
      return tanh(z);
    case 'relu':
      return relu(z);
    case 'sigmoid':
    default:
      return sigmoid(z);
  }
}

function activationDerivative(z: number, yHat: number, fn: ActivationFn): number {
  switch (fn) {
    case 'tanh':
      return 1 - yHat * yHat;
    case 'relu':
      return z > 0 ? 1 : 0;
    case 'sigmoid':
    default:
      return yHat * (1 - yHat);
  }
}

/**
 * Entrena un modelo de red neuronal muy simple (una sola neurona)
 * para clasificar niveles de satisfacción.
 *
 * Este ejercicio es educativo: muestra forward pass, error, gradiente
 * y actualización de pesos/bias con gradiente descendente.
 *
 * Novedades para el laboratorio:
 * - activación configurable (sigmoid/tanh/relu)
 * - history con snapshots (weights/bias/z/yHat) para animación y sliders
 * - dataset exportable para frontera de decisión y puntos 2D
 */
export function trainSatisfaction(options?: {
  learningRate?: number; // Tasa de aprendizaje
  epochs?: number; // Número de épocas
  logEvery?: number; // Cada cuántas épocas guardar un punto en history
  seed?: number; // Semilla opcional para reproducibilidad

  /** Activación para experimentar/enseñar (default: sigmoid). */
  activation?: ActivationFn;
}): TrainingResult {
  const {
    learningRate: learningRateRaw = 0.01,
    epochs: epochsRaw = 2_000, // default amigable para UI/visualizaciones
    logEvery: logEveryRaw = 100,
    seed,
    activation = 'sigmoid',
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

  // Dataset (exportable)
  // Nota didáctica: estos valores NO están normalizados.
  // Esto puede usarse como lección: escalas distintas afectan la magnitud de los pesos.
  const { X, y } = getSatisfactionDataset();

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
      const yHat = activationForward(z, activation);

      lastZ = z;
      lastYHat = yHat;

      // Error (MSE educativo)
      const error = y[i] - yHat;

      // Gradiente (cadena: error * f'(z))
      // En clasificación real suele usarse cross-entropy, pero MSE funciona para visualización educativa.
      const gradAct = activationDerivative(z, yHat, activation);
      const gradient = error * gradAct;

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
  const testZ = testInput[0] * w[0] + testInput[1] * w[1] + b;
  const pred = activationForward(testZ, activation);

  return {
    weights: [w[0], w[1]], // snapshot defensivo (evita mutaciones accidentales)
    bias: b,
    prediction: pred,
    history,
  };
}