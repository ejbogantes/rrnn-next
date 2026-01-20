// lib/nn-sales.ts

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
 * Dataset del ejercicio de ventas.
 * Lo exportamos para poder:
 * - dibujar los puntos en 2D
 * - mostrar frontera de decisión
 * - explicar separabilidad
 */
export function getSalesDataset(): { X: [number, number][], y: number[] } {
    const X: [number, number][] = [
        [2, 10],
        [3, 15],
        [5, 20],
        [1, 5],
        [4, 18],
        [0.5, 3],
    ];
    const y: number[] = [0, 1, 1, 0, 1, 0];
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

/**
 * Derivadas (en términos de salida cuando conviene):
 * - sigmoid': yHat * (1 - yHat)
 * - tanh': 1 - yHat^2
 * - relu': z > 0 ? 1 : 0
 */
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
 * Entrena un modelo neuronal extremadamente simple (una sola neurona)
 * para predecir la salida según dos características.
 *
 * Diseñado para visualización educativa:
 * - history muestreado con logEvery
 * - snapshots de weights/bias/z/yHat para animación y sliders
 * - activación configurable (sigmoid/tanh/relu) para comparación A/B
 */
export function trainSales(options?: {
    learningRate?: number;
    epochs?: number;
    logEvery?: number;
    seed?: number;

    /** Activación para experimentar/enseñar (default: sigmoid). */
    activation?: ActivationFn;
}): TrainingResult {
    const {
        learningRate: learningRateRaw = 0.01,
        epochs: epochsRaw = 2_000,
        logEvery: logEveryRaw = 100,
        seed,
        activation = 'sigmoid',
    } = options || {};

    // Validaciones suaves: evitan NaN / negativos / valores absurdos
    const learningRate =
        Number.isFinite(learningRateRaw) && learningRateRaw > 0 ? learningRateRaw : 0.01;

    const epochs = Number.isFinite(epochsRaw) ? Math.max(1, Math.floor(epochsRaw)) : 2_000;

    const logEvery = Number.isFinite(logEveryRaw)
        ? Math.max(1, Math.floor(logEveryRaw))
        : 100;

    // Determinismo opcional (sin tocar Math.random global)
    const rand = seed !== undefined ? mulberry32(seed) : Math.random;

    // Dataset
    const { X, y } = getSalesDataset();

    // Inicialización
    const w = [rand(), rand()];
    let b = rand();

    const history: TrainingPoint[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalError = 0;

        // Snapshot representativo (último sample)
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
            const gradAct = activationDerivative(z, yHat, activation);
            const gradient = error * gradAct;

            // Update
            w[0] += learningRate * gradient * x1;
            w[1] += learningRate * gradient * x2;
            b += learningRate * gradient;

            totalError += Math.pow(error, 2);
        }

        if (epoch % logEvery === 0) {
            history.push({
                epoch,
                error: totalError / X.length,
                weights: [w[0], w[1]],
                bias: b,
                z: lastZ,
                yHat: lastYHat,
            });
        }
    }

    // Evaluación final (ejemplo de prueba)
    const testInput: [number, number] = [3.5, 12];
    const testZ = testInput[0] * w[0] + testInput[1] * w[1] + b;
    const pred = activationForward(testZ, activation);

    return {
        weights: [w[0], w[1]],
        bias: b,
        prediction: pred,
        history,
    };
}