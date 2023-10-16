import { Neuron } from './Neuron';

export class NeuralNetwork {
    public layers: Neuron[][];

    constructor(layerSizes: number[]) {
        this.layers = [];

        console.log(`init-layerSizes: ${layerSizes}`);

        // Input layer
        this.layers.push(
            Array.from({ length: layerSizes[0] }, () => new Neuron(1))
        );

        // Hidden layers
        for (let i = 1; i < layerSizes.length; i++) {
            console.log(`hidden-layer-init_sizes: ${layerSizes[i]}`);
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
                console.log(
                    `fwdProp_ weights: ${neuron.weights}; bias: ${neuron.bias}`
                );
                const weightedSum = neuron.weights.reduce(
                    (sum: number, weight: number, index: number) =>
                        sum + weight * outputs[index],
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
        // console.log(`sigmoidInput: ${x}`);
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
            console.log(
                `TRAINING_ERR_CALC_ target: ${targets[i]} output: ${
                    outputs[i]
                } error: ${targets[i] - outputs[i]}`
            );
            outputErrors.push(targets[i] - outputs[i]);
        }

        // Backpropagation learning algorithm
        // update biases and weights for output layer
        for (let i = 0; i < this.layers[this.layers.length - 1].length; i++) {
            const neuron = this.layers[this.layers.length - 1][i];
            console.log(
                `TRAINING_ bias: ${neuron.bias} outputErrors: ${outputErrors[i]} LR: ${learningRate}`
            );
            neuron.bias +=
                outputErrors[i] *
                this.sigmoidDerivative(outputs[i]) *
                learningRate;
            for (let j = 0; j < neuron.weights.length; j++) {
                console.log(
                    `TRAINING_ weight: ${neuron.weights[j]} outputErrors: ${outputErrors[i]} inputs: ${inputs[j]} LR: ${learningRate}`
                );
                neuron.weights[j] +=
                    outputErrors[i] *
                    this.sigmoidDerivative(outputs[i]) *
                    inputs[j] *
                    learningRate;
            }
        }
        // update biases and weights for hidden layers
        for (let i = this.layers.length - 2; i > 0; i--) {
            for (const neuron of this.layers[i]) {
                let error = 0;
                for (let j = 0; j < this.layers[i + 1].length; j++) {
                    console.log(
                        `TRAINING_ bias: ${neuron.bias} outputErrors: ${outputErrors[i]} LR: ${learningRate}`
                    );
                    console.log(
                        `TRAINING_ weight: ${neuron.weights[j]} outputErrors: ${outputErrors[i]} inputs: ${inputs[j]} LR: ${learningRate}`
                    );
                    error +=
                        this.layers[i + 1][j].weights[
                            this.layers[i].indexOf(neuron)
                        ] * outputErrors[j];
                }
                error *= this.sigmoidDerivative(
                    neuron.weights.reduce(
                        (sum: number, weight: number, index: number) =>
                            sum + weight * inputs[index],
                        0
                    ) + neuron.bias
                );

                neuron.bias += error * learningRate;
                for (let j = 0; j < neuron.weights.length; j++) {
                    neuron.weights[j] += error * inputs[j] * learningRate;
                }
            }
        }
    }

    private sigmoidDerivative(x: number): number {
        // console.log(`derivativeOfSigmoidInput: ${x}`);
        return this.sigmoid(x) * (1 - this.sigmoid(x));
    }
}
