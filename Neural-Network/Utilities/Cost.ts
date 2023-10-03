enum CostType {
    MeanSquareError,
    CrossEntropy,
}

interface ICost {
    costFunction(predictedOutputs: number[], expectedOutputs: number[]): number;
    costDerivative(predictedOutput: number, expectedOutput: number): number;
    costFunctionType(): CostType;
}

class Cost {
    static getCostFromType(type: CostType): ICost {
        switch (type) {
            case CostType.MeanSquareError:
                return new MeanSquaredError();
            case CostType.CrossEntropy:
                return new CrossEntropy();
            default:
                console.error('Unhandled cost type');
                return new MeanSquaredError();
        }
    }
}

class MeanSquaredError implements ICost {
    costFunction(
        predictedOutputs: number[],
        expectedOutputs: number[]
    ): number {
        let cost = 0;
        for (let i = 0; i < predictedOutputs.length; i++) {
            const error = predictedOutputs[i] - expectedOutputs[i];
            cost += error * error;
        }
        return 0.5 * cost;
    }

    costDerivative(predictedOutput: number, expectedOutput: number): number {
        return predictedOutput - expectedOutput;
    }

    costFunctionType(): CostType {
        return CostType.MeanSquareError;
    }
}

class CrossEntropy implements ICost {
    costFunction(
        predictedOutputs: number[],
        expectedOutputs: number[]
    ): number {
        let cost = 0;
        for (let i = 0; i < predictedOutputs.length; i++) {
            const x = predictedOutputs[i];
            const y = expectedOutputs[i];
            const v = y === 1 ? -Math.log(x) : -Math.log(1 - x);
            cost += isNaN(v) ? 0 : v;
        }
        return cost;
    }

    costDerivative(predictedOutput: number, expectedOutput: number): number {
        const x = predictedOutput;
        const y = expectedOutput;
        if (x === 0 || x === 1) {
            return 0;
        }
        return (-x + y) / (x * (x - 1));
    }

    costFunctionType(): CostType {
        return CostType.CrossEntropy;
    }
}
