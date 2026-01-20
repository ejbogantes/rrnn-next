// app/api/train/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { trainSatisfaction } from '@/lib/nn-satisfaction';
import { trainSales } from '@/lib/nn-sales';
import type { ActivationFn, ExperimentConfig, ExperimentResult, TrainMeta } from '@/lib/types';

/**
 * API: /api/train
 *
 * Endpoint educativo para entrenar modelos neuronales simples.
 * Permite experimentar con hiperparámetros y observar el proceso de aprendizaje
 * de forma controlada y reproducible.
 *
 * ✅ Nuevo:
 * - soporte de activation: sigmoid | tanh | relu
 * - respuesta "modo laboratorio" (experiment: { config, meta, result })
 * - mantiene compatibilidad con el response anterior (model/hyperparameters/result)
 */
export async function GET(req: NextRequest) {
    const { searchParams } = new URL(req.url);

    // --- Validación y saneamiento de parámetros ---

    // Modelo permitido (evita valores arbitrarios)
    const allowedModels = new Set(['sales', 'satisfaction']);
    const modelParam = searchParams.get('model');
    const model = allowedModels.has(modelParam || '')
        ? (modelParam as 'sales' | 'satisfaction')
        : 'satisfaction';

    // Activación permitida
    const allowedActivations = new Set<ActivationFn>(['sigmoid', 'tanh', 'relu']);
    const activationParam = (searchParams.get('activation') || 'sigmoid') as ActivationFn;
    const activation: ActivationFn = allowedActivations.has(activationParam)
        ? activationParam
        : 'sigmoid';

    // Epochs: límites razonables para UX (evita bloquear la UI)
    const epochsRaw = parseInt(searchParams.get('epochs') || '2000', 10);
    const epochs = Number.isFinite(epochsRaw) ? Math.min(Math.max(1, epochsRaw), 50_000) : 2000;

    // Learning rate: debe ser positivo y finito
    const learningRateRaw = parseFloat(searchParams.get('learningRate') || '0.01');
    const learningRate =
        Number.isFinite(learningRateRaw) && learningRateRaw > 0 ? learningRateRaw : 0.01;

    // Seed opcional para reproducibilidad (si no es válido, se ignora)
    const seedRaw = searchParams.get('seed');
    const seed =
        seedRaw !== null && !Number.isNaN(parseInt(seedRaw, 10)) ? parseInt(seedRaw, 10) : undefined;

    // Control de densidad del history (ideal para sliders y animaciones)
    // Ej: ~200 puntos máximo en la gráfica
    const logEvery = Math.max(1, Math.floor(epochs / 200));

    // Config del experimento (para export/import, A/B, etc.)
    const config: ExperimentConfig = {
        model,
        epochs,
        learningRate,
        activation,
        seed,
    };

    // Meta del experimento (lo que realmente se usó)
    const meta: TrainMeta = {
        epochs,
        learningRate,
        activation,
        seed,
        logEvery,
    };

    try {
        let result;

        // --- Entrenamiento según el modelo seleccionado ---
        switch (model) {
            case 'sales':
                result = trainSales({
                    epochs,
                    learningRate,
                    seed,
                    logEvery,
                    activation,
                });
                break;

            case 'satisfaction':
            default:
                result = trainSatisfaction({
                    epochs,
                    learningRate,
                    seed,
                    logEvery,
                    activation,
                });
                break;
        }

        const experiment: ExperimentResult = {
            config,
            meta,
            result,
        };

        // --- Respuesta enriquecida (didáctica) ---
        // Mantiene compatibilidad con la UI existente:
        // - model/hyperparameters/result (antiguo)
        // + experiment (nuevo)
        return NextResponse.json(
            {
                // ✅ Compatibilidad con tu page.tsx actual
                model,
                hyperparameters: {
                    epochs,
                    learningRate,
                    activation,
                    seed,
                    logEvery,
                },
                result,

                // ✅ Nuevo: “modo laboratorio”
                experiment,
            },
            {
                // Evita caching accidental (importante en demos)
                headers: { 'Cache-Control': 'no-store' },
            }
        );
    } catch (error: unknown) {
        if (error instanceof Error) {
            console.error('Error al entrenar el modelo:', error.message);
        } else {
            console.error('Error desconocido durante el entrenamiento:', error);
        }

        return NextResponse.json(
            { error: 'Error durante el entrenamiento del modelo.' },
            { status: 500, headers: { 'Cache-Control': 'no-store' } }
        );
    }
}