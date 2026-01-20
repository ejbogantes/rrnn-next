// lib/types.ts

/* =========================================================
   Tipos base de entrenamiento (YA EXISTENTES, SE MANTIENEN)
   ========================================================= */

/**
 * Representa un punto de entrenamiento en la evolución del modelo.
 * Se utiliza para registrar el estado del aprendizaje en una época concreta
 * y así poder visualizar cómo cambia la red neuronal con el tiempo.
 */
export interface TrainingPoint {
    /** Número de época (iteración completa sobre el dataset). */
    epoch: number;

    /** Error promedio de la época (por ejemplo, MSE o BCE). */
    error: number;

    /** Pesos del modelo en esta época. */
    weights?: number[];

    /** Sesgo (bias) del modelo en esta época. */
    bias?: number;

    /** Valor previo a la activación: z = w·x + b. */
    z?: number;

    /** Salida del modelo después de la activación (ŷ). */
    yHat?: number;
}

/**
 * Resultado completo del entrenamiento de una red neuronal simple.
 */
export interface TrainingResult {
    weights: number[];
    bias: number;
    prediction: number;
    history: TrainingPoint[];
}

/* =========================================================
   🔥 NUEVO: conceptos de laboratorio / experimento
   ========================================================= */

/**
 * Funciones de activación soportadas por el laboratorio.
 * Esto habilita comparación Sigmoid vs ReLU vs Tanh.
 */
export type ActivationFn = 'sigmoid' | 'relu' | 'tanh';

/**
 * Configuración completa de un experimento.
 * Esto es lo que se puede:
 * - comparar (A/B)
 * - guardar
 * - volver a cargar
 */
export interface ExperimentConfig {
    model: 'sales' | 'satisfaction';

    /** Hiperparámetros */
    epochs: number;
    learningRate: number;

    /** Activación seleccionada */
    activation: ActivationFn;

    /** Seed opcional para reproducibilidad */
    seed?: number;
}

/**
 * Metadatos derivados del entrenamiento.
 * No afectan el modelo, pero explican lo que pasó.
 */
export interface TrainMeta {
    epochs: number;
    learningRate: number;
    activation: ActivationFn;
    seed?: number;

    /** Cada cuántas épocas se guardó history */
    logEvery: number;
}

/**
 * Resultado completo de un experimento ejecutado.
 * Esto es lo que usa el frontend para:
 * - timeline
 * - visualizaciones
 * - explicación viva
 */
export interface ExperimentResult {
    config: ExperimentConfig;
    meta: TrainMeta;
    result: TrainingResult;
}

/**
 * Experimento A/B: dos corridas comparables.
 * Ej: mismo seed, distinto learning rate o activación.
 */
export interface ExperimentComparison {
    left: ExperimentResult;
    right: ExperimentResult;
}

/* =========================================================
   🔬 Tipos para explicación viva / interpretación
   ========================================================= */

/**
 * Cambio detectado entre dos puntos de entrenamiento.
 * Se usa para explicar "qué está aprendiendo".
 */
export interface ParameterDelta {
    weightIndex: number;
    delta: number;
}

/**
 * Análisis simple de una época (interpretabilidad).
 */
export interface TrainingInsight {
    epoch: number;

    /** Cambios relevantes en pesos */
    weightDeltas: ParameterDelta[];

    /** Cambio en bias */
    biasDelta: number;

    /** Indica posible saturación de la activación */
    saturated?: boolean;

    /** Mensaje pedagógico para el alumno */
    message: string;
}