// lib/types.ts

/**
 * Representa un punto de entrenamiento en la evolución del modelo.
 * Se utiliza para registrar el error promedio de cada época (epoch),
 * permitiendo visualizar el proceso de aprendizaje del modelo.
 */
export interface TrainingPoint {
    /**
     * Número de época (iteración de entrenamiento).
     */
    epoch: number;

    /**
     * Error promedio de la época (por ejemplo, MSE o BCE).
     */
    error: number;
}

/**
 * Resultado completo del entrenamiento de una red neuronal simple.
 * Contiene los pesos finales, el sesgo, la predicción de prueba
 * y el historial del error durante el entrenamiento.
 */
export interface TrainingResult {
    /**
     * Pesos finales aprendidos por el modelo (uno por cada variable de entrada).
     */
    weights: number[];

    /**
     * Sesgo (bias) aprendido durante el entrenamiento.
     */
    bias: number;

    /**
     * Predicción final del modelo sobre un ejemplo de prueba.
     */
    prediction: number;

    /**
     * Historial de puntos de entrenamiento, usado para graficar la evolución del error.
     */
    history: TrainingPoint[];
}