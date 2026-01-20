// app/api/train/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { trainSatisfaction } from '@/lib/nn-satisfaction';
import { trainSales } from '@/lib/nn-sales';

/**
 * API: /api/train
 *
 * Endpoint educativo para entrenar modelos neuronales simples.
 * Permite experimentar con hiperparámetros y observar el proceso de aprendizaje
 * de forma controlada y reproducible.
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

    // Epochs: límites razonables para UX (evita bloquear la UI)
    const epochsRaw = parseInt(searchParams.get('epochs') || '2000', 10);
    const epochs = Number.isFinite(epochsRaw)
        ? Math.min(Math.max(1, epochsRaw), 50_000)
        : 2000;

    // Learning rate: debe ser positivo y finito
    const learningRateRaw = parseFloat(searchParams.get('learningRate') || '0.01');
    const learningRate =
        Number.isFinite(learningRateRaw) && learningRateRaw > 0
            ? learningRateRaw
            : 0.01;

    // Seed opcional para reproducibilidad (si no es válido, se ignora)
    const seedRaw = searchParams.get('seed');
    const seed = seedRaw !== null && !Number.isNaN(parseInt(seedRaw, 10))
        ? parseInt(seedRaw, 10)
        : undefined;

    // Control de densidad del history (ideal para sliders y animaciones)
    // Ej: ~200 puntos máximo en la gráfica
    const logEvery = Math.max(1, Math.floor(epochs / 200));

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
                });
                break;

            case 'satisfaction':
            default:
                result = trainSatisfaction({
                    epochs,
                    learningRate,
                    seed,
                    logEvery,
                });
                break;
        }

        // --- Respuesta enriquecida (didáctica) ---
        // No solo devolvemos el resultado, sino también el contexto del entrenamiento.
        return NextResponse.json({
            model,
            hyperparameters: {
                epochs,
                learningRate,
                seed,
                logEvery,
            },
            result,
        });
    } catch (error: unknown) {
        if (error instanceof Error) {
            console.error('Error al entrenar el modelo:', error.message);
        } else {
            console.error('Error desconocido durante el entrenamiento:', error);
        }

        return NextResponse.json(
            { error: 'Error durante el entrenamiento del modelo.' },
            { status: 500 }
        );
    }
}