class NeuralNetwork {
    readonly layers: Layer[];
    readonly layerSizes: number[];
    cost: ICost;
    rng: MathRandom;
    batchLearnData: NetworkLearnData[];

    constructor(...layerSizes: number[]) {
        this.layerSizes = layerSizes;
        this.rng = new MathRandom();

        this.layers = [];
        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            this.layers.push(
                new Layer(this.layerSizes[i], this.layerSizes[i + 1], this.rng)
            );
        }

        this.cost = new MeanSquaredError();
    }

    classify(inputs: number[]): [number, number[]] {
        const outputs = this.calculateOutputs(inputs);
        const predicatedClass = this.maxValueIndex(outputs);
        return [predicatedClass, outputs];
    }

    calculateOutputs(inputs: number[]): number[] {
        for (const layer of this.layers) {
            inputs = layer.calculateOutputs(inputs);
        }
        return inputs;
    }

    Learn(
        trainingData: DataPoint[],
        learnRate: number,
        regularization: number = 0,
        momentum: number = 0
    ): void {
        if (
            !this.batchLearnData ||
            this.batchLearnData.length !== trainingData.length
        ) {
            this.batchLearnData = trainingData.map(
                () => new NetworkLearnData(this.layers)
            );

            trainingData.forEach((data, i) => {
                this.updateGradients(data, this.batchLearnData[i]);
            });

            const batchSize = trainingData.length;
            for (let i = 0; i < this.layers.length; i++) {
                this.layers[i].applyGradients(
                    learnRate / batchSize,
                    regularization,
                    momentum
                );
            }
        }
    }

    updateGradients(data: DataPoint, learnData: NetworkLearnData): void {
        let inputsToNextLayer: number[] = data.inputs;

        for (let i = 0; i < this.layers.length; i++) {
            inputsToNextLayer = this.layers[i].calculateOutputs(
                inputsToNextLayer,
                learnData.layerData[i]
            );
        }

        const outputLayerIndex = this.layers.length - 1;
        const outputLayer = this.layers[outputLayerIndex];
        const outputLearnData = learnData.layerData(outputLayerIndex);

        outputLayer.calculateOutputLayerNodeValues(
            outputLearnData,
            data.expectedOutputs,
            this.cost
        );
        outputLayer.updateGradients(outputLearnData);

        for (let i = outputLayerIndex - 1; i >= 0; i--) {
            const layerLearnData = learnData.layerData[i];
            const hiddenLayer = this.layers[i];
            hiddenLayer.calculateHiddenLayerNodeValues(
                layerLearnData,
                this.layers[i + 1],
                learnData.layerData[i + 1].nodeValues
            );
            hiddenLayer.updateGradients(layerLearnData);
        }
    }

    setCostFunction(costFunction: ICost): void {
        this.cost = costFunction;
    }

    setActivationFunction(activation: IActivation): void {
        this.setActivationFunction(activation, activation);
    }

    setActivationFunction(
        activation: IActivation,
        outputLayerActivation: IActivation
    ): void {
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.layers[i].setActivationFunction(activation);
        }
        this.layers[this.layers.length - 1].setActivationFunction(
            outputLayerActivation
        );
    }

    maxValueIndex(values: number[]): number {
        let maxValue = Number.MIN_VALUE;
        let index = 0;
        for (let i = 0; i < values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                index = i;
            }
        }
        return index;
    }
}

interface DataPoint {
    inputs: number[];
    expectedOutputs: number[];
}

class NetworkLearnData {
    layerData: LayerLearnData[];

    constructor(layers: Layer[]) {
        this.layerData = layers.map(layer => new LayerLearnData(layer));
    }
}

class LayerLearnData {
    inputs: number[];
    weightedInputs: number[];
    activations: number[];
    nodeValues: number[];

    constructor(layer: Layer) {
        this.weightedInputs = new Array(layer.numNodesOut).fill(0);
        this.activations = new Array(layer.numNodesOut).fill(0);
        this.nodeValues = new Array(layer.numNodesOut).fill(0);
    }
}
class MathRandom {
    nextDouble(): number {
        return Math.random();
    }
}
