// lib/types.ts

/**
 * Representa un punto de entrenamiento en la evolución del modelo.
 * Se utiliza para registrar el estado del aprendizaje en una época concreta
 * y así poder visualizar cómo cambia la red neuronal con el tiempo.
 */
export interface TrainingPoint {
    /**
     * Número de época (iteración completa sobre el dataset).
     */
    epoch: number;

    /**
     * Error promedio de la época (por ejemplo, MSE o BCE).
     * Es una consecuencia del estado actual del modelo.
     */
    error: number;

    /**
     * Pesos del modelo en esta época.
     * Opcional para no romper ejercicios simples que solo grafican el error.
     */
    weights?: number[];

    /**
     * Sesgo (bias) del modelo en esta época.
     * Permite visualizar cómo se desplaza el umbral de activación.
     */
    bias?: number;

    /**
     * (Opcional) Valor previo a la activación: z = w·x + b.
     * Muy útil para visualizaciones didácticas del forward pass.
     */
    z?: number;

    /**
     * (Opcional) Salida del modelo después de la activación (ŷ).
     * Permite mostrar cómo cambia la predicción a lo largo del entrenamiento.
     */
    yHat?: number;
}

/**
 * Resultado completo del entrenamiento de una red neuronal simple.
 * Contiene el estado final del modelo y el historial de aprendizaje,
 * pensado tanto para evaluación como para visualización educativa.
 */
export interface TrainingResult {
    /**
     * Pesos finales aprendidos por el modelo (uno por cada variable de entrada).
     */
    weights: number[];

    /**
     * Sesgo (bias) final aprendido durante el entrenamiento.
     */
    bias: number;

    /**
     * Predicción final del modelo sobre un ejemplo de prueba.
     */
    prediction: number;

    /**
     * Historial de puntos de entrenamiento.
     * Puede usarse para:
     * - graficar el error
     * - recorrer épocas con un slider
     * - animar cómo cambian pesos, bias y activaciones
     */
    history: TrainingPoint[];
}