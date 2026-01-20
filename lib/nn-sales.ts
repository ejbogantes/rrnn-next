// lib/nn-sales.ts

// Importamos los tipos de datos que definen la estructura del resultado y del historial.
// Esto permite mantener consistencia entre distintos modelos (ventas, satisfacción, etc.).
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
 * Entrena un modelo neuronal extremadamente simple (una sola neurona)
 * para predecir la probabilidad de éxito en ventas según características básicas.
 *
 * Este ejemplo está diseñado con fines educativos,
 * explicando paso a paso el flujo de entrenamiento con gradiente descendente.
 *
 * @param options Configuración opcional del entrenamiento.
 * @returns TrainingResult con pesos, sesgo, predicción y evolución del error.
 */
export function trainSales(options?: {
    learningRate?: number; // Tasa de aprendizaje (cuánto ajustamos los pesos cada paso)
    epochs?: number;       // Número total de iteraciones del entrenamiento
    logEvery?: number;     // Cada cuántas épocas se guarda info en el historial
    seed?: number;         // Semilla opcional para reproducibilidad
}): TrainingResult {
    // --- Parámetros configurables con valores por defecto ---
    // Nota pedagógica: para visualizaciones interactivas, 15k por defecto suele ser mucho.
    // Dejamos un default más ligero y el caller puede pedir más si lo necesita.
    const {
        learningRate: learningRateRaw = 0.01,
        epochs: epochsRaw = 2_000,
        logEvery: logEveryRaw = 100,
        seed,
    } = options || {};

    // Validaciones suaves: evitan NaN / negativos / valores absurdos que rompan la UI.
    const learningRate =
        Number.isFinite(learningRateRaw) && learningRateRaw > 0 ? learningRateRaw : 0.01;

    const epochs = Number.isFinite(epochsRaw) ? Math.max(1, Math.floor(epochsRaw)) : 2_000;

    const logEvery = Number.isFinite(logEveryRaw)
        ? Math.max(1, Math.floor(logEveryRaw))
        : 100;

    // --- Determinismo opcional ---
    // Si hay seed, usamos un PRNG local determinista; si no, usamos Math.random normal.
    const rand = seed !== undefined ? mulberry32(seed) : Math.random;

    // --- Funciones auxiliares ---
    // Sigmoide: función de activación que transforma valores numéricos en [0, 1].
    const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

    // Derivada de la sigmoide: usada para calcular el gradiente.
    // Nota: aquí la derivada se expresa en función de yHat para ahorrar cómputo.
    const sigmoidDerivative = (yHat: number): number => yHat * (1 - yHat);

    // --- Datos de entrenamiento ---
    // X representa pares [presupuesto, leads] o cualquier variable de entrada.
    // y indica si la venta fue exitosa (1) o no (0).
    const X: number[][] = [
        [2, 10],
        [3, 15],
        [5, 20],
        [1, 5],
        [4, 18],
        [0.5, 3],
    ];

    const y: number[] = [0, 1, 1, 0, 1, 0];

    // --- Inicialización de parámetros ---
    // Pesos iniciales aleatorios. Se usa const porque la referencia no cambia.
    const w = [rand(), rand()];

    // Sesgo (bias), ajustable durante el entrenamiento.
    let b = rand();

    // Historial para graficar / animar la evolución.
    const history: TrainingPoint[] = [];

    // --- Bucle de entrenamiento principal ---
    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalError = 0; // Error acumulado de la época.

        // Guardamos un "snapshot" representativo por época (del último sample visto).
        // Esto es útil para visualización (z/yHat) sin inflar el history por cada sample.
        let lastZ = 0;
        let lastYHat = 0;

        // Recorremos todos los ejemplos de entrenamiento.
        for (let i = 0; i < X.length; i++) {
            // Desestructuramos las dos variables de entrada (x1, x2).
            const [x1, x2] = X[i];

            // Paso 1: Forward pass (propagación hacia adelante)
            // Combinamos las entradas ponderadas más el sesgo.
            const z = x1 * w[0] + x2 * w[1] + b;

            // Aplicamos la función sigmoide → salida entre 0 y 1.
            const yHat = sigmoid(z);

            // Snapshot para visualización
            lastZ = z;
            lastYHat = yHat;

            // Paso 2: Cálculo del error
            // Diferencia entre la salida real y la predicha.
            const error = y[i] - yHat;

            // Paso 3: Gradiente descendente
            // Nota educativa: estamos usando MSE con sigmoide por simplicidad visual.
            // (En clasificación, típicamente se usa cross-entropy, pero MSE sirve para un demo.)
            const gradient = error * sigmoidDerivative(yHat);

            // Paso 4: Actualización de los pesos y el sesgo
            w[0] += learningRate * gradient * x1;
            w[1] += learningRate * gradient * x2;
            b += learningRate * gradient;

            // Acumulamos el error cuadrático medio (MSE parcial)
            totalError += Math.pow(error, 2);
        }

        // Cada logEvery épocas, guardamos el error promedio + estado del modelo.
        // Esto permite:
        // - graficar error
        // - slider por epoch
        // - animar cambio de pesos/bias/activación
        if (epoch % logEvery === 0) {
            history.push({
                epoch,
                error: totalError / X.length,
                weights: [w[0], w[1]], // snapshot (evita mutación accidental)
                bias: b,
                z: lastZ,
                yHat: lastYHat,
            });
        }
    }

    // --- Evaluación final ---
    // Probamos el modelo con un ejemplo nuevo.
    // Nota: si luego normalizas X, este input también debe normalizarse.
    const testInput: [number, number] = [3.5, 12];
    const pred = sigmoid(testInput[0] * w[0] + testInput[1] * w[1] + b);

    // --- Resultado ---
    // Retornamos el modelo entrenado y los datos del entrenamiento.
    return {
        weights: [w[0], w[1]], // snapshot final (evita exponer referencia mutable)
        bias: b,
        prediction: pred,
        history,
    };
}