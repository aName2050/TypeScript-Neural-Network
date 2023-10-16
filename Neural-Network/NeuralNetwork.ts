class NeuralNetwork {
    public layers: Neuron[][];

    constructor(layerSizes: number[]) {
        this.layers = [];

        // Input layer
        this.layers.push(
            Array.from({ length: layerSizes[0] }, () => new Neuron(1))
        );

        // Hidden layers
        for (let i = 0; i < layerSizes.length - 1; i++) {
            this.layers.push(
                Array.from(
                    { length: layerSizes[i] },
                    () => new Neuron(layerSizes[i - 1])
                )
            );
        }

        // Output Layer
        this.layers.push(
            Array.from(
                { length: layerSizes[layerSizes.length - 1] },
                () => new Neuron(layerSizes[layerSizes.length - 2])
            )
        );
    }

    public forwardPropagation(inputs: number[]): number[] {
        let outputs = inputs;

        for (const layer of this.layers) {
            const newOutputs = [];
            for (const neuron of layer) {
                const weightedSum = neuron.weights.reduce(
                    (sum, weight, index) => sum + weight * outputs[index],
                    0
                );
                const activation = this.sigmoid(weightedSum + neuron.bias);
                newOutputs.push(activation);
            }
            outputs = newOutputs;
        }

        return outputs;
    }

    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    public train(
        inputs: number[],
        targets: number[],
        learningRate: number = 0.1
    ): void {
        // Forward propagation
        const outputs = this.forwardPropagation(inputs);

        // Error calculation per neuron
        const outputErrors = [];
        for (let i = 0; i < outputs.length; i++) {
            outputErrors.push(targets[i] - outputs[i]);
        }

        // Backpropagation learning algorithm
        // update biases and weights for output layer
        for (let i = 0; i < this.layers[this.layers.length - 1].length; i++) {
            const neuron = this.layers[this.layers.length - 1][i];
            neuron.bias +=
                outputErrors[i] *
                this.sigmoidDerivative(outputs[i]) *
                learningRate;
            for (let j = 0; j < neuron.weights.length; j++) {
                neuron.weights;
            }
        }
    }

    private sigmoidDerivative(x: number): number {
        return this.sigmoid(x) * (1 - this.sigmoid(x));
    }
}
