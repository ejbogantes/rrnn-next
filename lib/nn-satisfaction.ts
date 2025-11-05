// lib/nn-satisfaction.ts

// Importamos los tipos que definen la estructura de los resultados y puntos de entrenamiento.
// Esto mejora la legibilidad, el autocompletado y la seguridad de tipos.
import { TrainingResult, TrainingPoint } from './types';

/**
 * Entrena un modelo de red neuronal muy simple (una sola neurona)
 * para clasificar niveles de satisfacción.
 *
 * Este ejercicio tiene fines educativos y explica paso a paso
 * cómo funciona el aprendizaje por gradiente en una red básica.
 *
 * @param options Configuración del modelo y entrenamiento.
 * @returns TrainingResult con pesos, sesgo, predicción y evolución del error.
 */
export function trainSatisfaction(options?: {
  learningRate?: number; // Tasa de aprendizaje
  epochs?: number;        // Número de iteraciones
  logEvery?: number;      // Cada cuántos pasos guardar el error en el historial
  seed?: number;          // Semilla opcional para resultados reproducibles
}): TrainingResult {

  // Desestructuramos los parámetros con valores por defecto.
  const {
    learningRate = 0.01,
    epochs = 10_000,
    logEvery = 100,
    seed,
  } = options || {};

  // --- Determinismo opcional (semilla) ---
  // Si se define una semilla, reemplazamos Math.random() con una versión determinista.
  // Esto permite obtener los mismos resultados en cada ejecución, útil en entornos educativos.
  if (seed !== undefined) {
    let s = seed;
    Math.random = () => {
      const x = Math.sin(s++) * 10_000;
      return x - Math.floor(x);
    };
  }

  // --- Funciones auxiliares ---
  // La función sigmoide transforma valores numéricos en un rango [0, 1].
  const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

  // La derivada de la sigmoide se utiliza para calcular el gradiente durante el aprendizaje.
  const sigmoidDerivative = (yHat: number): number => yHat * (1 - yHat);

  // --- Datos de entrenamiento ---
  // X contiene pares [horas de atención, nivel de servicio], ambos normalizados.
  // y contiene las etiquetas: 1 = cliente satisfecho, 0 = insatisfecho.
  const X: number[][] = [
    [2, 5],
    [10, 4],
    [25, 2],
    [30, 1],
    [5, 4],
  ];

  const y: number[] = [1, 1, 0, 0, 1];

  // --- Inicialización de parámetros ---
  // Pesos iniciales (dos valores, uno por cada variable de entrada).
  // Se declaran como const porque la referencia al array no cambia.
  const w = [Math.random(), Math.random()];

  // Sesgo (bias), un valor que ajusta la salida del modelo.
  // Se actualiza durante el entrenamiento, por eso es let.
  let b = Math.random();

  // Historial de errores, para visualizar cómo aprende el modelo.
  const history: TrainingPoint[] = [];

  // --- Entrenamiento principal ---
  // Bucle principal: se repite durante 'epochs' iteraciones.
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalError = 0; // Error acumulado de la época actual.

    // Recorremos todos los ejemplos de entrenamiento (cada fila de X).
    for (let i = 0; i < X.length; i++) {
      // Descomponemos las variables de entrada.
      const [x1, x2] = X[i];

      // Paso 1: Forward pass
      // Calculamos la salida (z) combinando pesos, entradas y sesgo.
      const z = x1 * w[0] + x2 * w[1] + b;

      // Aplicamos la función sigmoide para obtener una predicción entre 0 y 1.
      const yHat = sigmoid(z);

      // Paso 2: Cálculo del error
      // La diferencia entre la etiqueta real (y) y la predicción (yHat).
      const error = y[i] - yHat;

      // Paso 3: Cálculo del gradiente
      // Este valor mide cómo debe ajustarse cada peso para reducir el error.
      const gradient = error * sigmoidDerivative(yHat);

      // Paso 4: Actualización de parámetros (Regla del gradiente descendente)
      w[0] += learningRate * gradient * x1; // Ajuste del peso 1
      w[1] += learningRate * gradient * x2; // Ajuste del peso 2
      b += learningRate * gradient;         // Ajuste del sesgo

      // Acumulamos el error cuadrático (MSE parcial).
      totalError += Math.pow(error, 2);
    }

    // Cada cierto número de épocas (logEvery), guardamos el error promedio.
    if (epoch % logEvery === 0) {
      history.push({
        epoch,
        error: totalError / X.length,
      });
    }
  }

  // --- Evaluación final ---
  // Probamos el modelo con una nueva entrada no vista durante el entrenamiento.
  const testInput = [12, 3];
  const pred = sigmoid(testInput[0] * w[0] + testInput[1] * w[1] + b);

  // --- Resultado ---
  // Retornamos un objeto con toda la información relevante del entrenamiento.
  return {
    weights: w,       // Pesos finales aprendidos
    bias: b,          // Sesgo final
    prediction: pred, // Predicción sobre el ejemplo de prueba
    history,          // Evolución del error
  };
}