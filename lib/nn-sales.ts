// lib/nn-sales.ts

// Importamos los tipos de datos que definen la estructura del resultado y del historial.
// Esto permite mantener consistencia entre distintos modelos (ventas, satisfacción, etc.).
import { TrainingResult, TrainingPoint } from './types';

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
    epochs?: number;        // Número total de iteraciones del entrenamiento
    logEvery?: number;      // Cada cuántas épocas se guarda el error en el historial
    seed?: number;          // Semilla opcional para reproducibilidad
}): TrainingResult {

    // --- Parámetros configurables con valores por defecto ---
    const {
        learningRate = 0.01,
        epochs = 15_000,
        logEvery = 100,
        seed,
    } = options || {};

    // --- Determinismo opcional ---
    // Si se proporciona una semilla, reemplazamos Math.random por una función determinista.
    if (seed !== undefined) {
        let s = seed;
        Math.random = () => {
            const x = Math.sin(s++) * 10_000;
            return x - Math.floor(x);
        };
    }

    // --- Funciones auxiliares ---
    // Sigmoide: función de activación que transforma valores numéricos en [0, 1].
    const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

    // Derivada de la sigmoide: usada para calcular el gradiente.
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
    const w = [Math.random(), Math.random()];

    // Sesgo (bias), ajustable durante el entrenamiento.
    let b = Math.random();

    // Historial del error para graficar la evolución.
    const history: TrainingPoint[] = [];

    // --- Bucle de entrenamiento principal ---
    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalError = 0; // Error acumulado de la época.

        // Recorremos todos los ejemplos de entrenamiento.
        for (let i = 0; i < X.length; i++) {
            // Desestructuramos las dos variables de entrada (x1, x2).
            const [x1, x2] = X[i];

            // Paso 1: Forward pass (propagación hacia adelante)
            // Combinamos las entradas ponderadas más el sesgo.
            const z = x1 * w[0] + x2 * w[1] + b;

            // Aplicamos la función sigmoide → salida entre 0 y 1.
            const yHat = sigmoid(z);

            // Paso 2: Cálculo del error
            // Diferencia entre la salida real y la predicha.
            const error = y[i] - yHat;

            // Paso 3: Gradiente descendente
            // Calculamos cuánto debe ajustarse cada parámetro.
            const gradient = error * sigmoidDerivative(yHat);

            // Paso 4: Actualización de los pesos y el sesgo
            w[0] += learningRate * gradient * x1;
            w[1] += learningRate * gradient * x2;
            b += learningRate * gradient;

            // Acumulamos el error cuadrático medio (MSE parcial)
            totalError += Math.pow(error, 2);
        }

        // Cada logEvery épocas, guardamos el error promedio.
        if (epoch % logEvery === 0) {
            history.push({
                epoch,
                error: totalError / X.length,
            });
        }
    }

    // --- Evaluación final ---
    // Probamos el modelo con un ejemplo nuevo.
    const testInput = [3.5, 12];
    const pred = sigmoid(testInput[0] * w[0] + testInput[1] * w[1] + b);

    // --- Resultado ---
    // Retornamos el modelo entrenado y los datos del entrenamiento.
    return {
        weights: w,       // Pesos finales aprendidos
        bias: b,          // Sesgo final
        prediction: pred, // Predicción sobre el ejemplo de prueba
        history,          // Evolución del error
    };
}