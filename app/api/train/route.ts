import { NextRequest, NextResponse } from 'next/server';
import { trainSatisfaction } from '@/lib/nn-satisfaction';
import { trainSales } from '@/lib/nn-sales';

/**
 * API: /api/train
 *
 * Este endpoint entrena un modelo neuronal simple (ventas o satisfacción)
 * con los parámetros configurables enviados desde el frontend.
 *
 * Permite al usuario experimentar con distintos hyperparámetros (epochs, learningRate)
 * y observar cómo cambia el proceso de aprendizaje.
 */
export async function GET(req: NextRequest) {
    const { searchParams } = new URL(req.url);
    const model = searchParams.get('model');
    const epochs = parseInt(searchParams.get('epochs') || '10000', 10);
    const learningRate = parseFloat(searchParams.get('learningRate') || '0.01');
    const seed = parseInt(searchParams.get('seed') || '42', 10);

    let result;

    try {
        switch (model) {
            case 'sales':
                result = trainSales({ epochs, learningRate, seed });
                break;
            case 'satisfaction':
            default:
                result = trainSatisfaction({ epochs, learningRate, seed });
                break;
        }

        return NextResponse.json(result);
    } catch (error: unknown) {
        if (error instanceof Error) {
            console.error('Error al entrenar el modelo:', error.message);
        } else {
            console.error('Error desconocido:', error);
        }

        return NextResponse.json(
            { error: 'Error durante el entrenamiento del modelo.' },
            { status: 500 }
        );
    }
}