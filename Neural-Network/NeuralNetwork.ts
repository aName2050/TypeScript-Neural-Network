import { Neuron } from './Neuron';

export class NeuralNetwork {
    public layers: Neuron[][];

    constructor(layerSizes: number[]) {
        this.layers = [];

        console.log(`INIT_ layerSizes: ${layerSizes}`);
        console.log(`INIT_ layers: ${layerSizes.length}`);

        // Input layer
        let inputLayerNeurons: Neuron[] = [];
        for (let i = 0; i < layerSizes[0]; i++) {
            console.log(
                `INIT-IN_ new neuron: ${i + 1}/${layerSizes[0]} inputs: 1`
            );

            inputLayerNeurons.push(new Neuron(1));
        }
        this.layers.push(inputLayerNeurons);

        // Hidden layer(s)
        for (let i = 1; i < layerSizes.length - 1; i++) {
            let layerNeurons: Neuron[] = [];
            for (let j = 0; j < layerSizes[i]; j++) {
                console.log(
                    `INIT-HL_ layer: ${i}/${
                        layerSizes.length - 2
                    } new neuron: ${j + 1}/${layerSizes[i]} inputs: ${
                        layerSizes[i - 1]
                    }`
                );
                layerNeurons.push(new Neuron(layerSizes[i - 1]));
            }
            this.layers.push(layerNeurons);
        }

        // Output layer
        let outputLayerNeurons: Neuron[] = [];
        for (let i = 0; i < layerSizes[layerSizes.length - 1]; i++) {
            console.log(
                `INIT-OUT_ new neuron: ${i + 1}/${
                    layerSizes[layerSizes.length - 1]
                } inputs: ${layerSizes.length - 2}`
            );
            outputLayerNeurons.push(new Neuron(layerSizes.length - 2));
        }
        this.layers.push(outputLayerNeurons);
    }

    /**
     * Forward Propagation Algorithm
     */
    public forwardPropagation(inputs: number[]): number[] {
        let outputs: number[] = inputs;

        for (const layer of this.layers) {
            const newOutputs: number[] = [];
            for (const neuron of layer) {
                console.log(
                    `FWD-PROP_ weights: ${neuron.weights} bias: ${neuron.bias}`
                );
                let weightedSum: number = 0;
                for (let i = 0; i < neuron.weights.length; i++) {
                    console.log(
                        `FWD-PROP_ weightedSum-equation: (${i + 1}/${
                            neuron.weights.length
                        }) ${weightedSum} + ${neuron.weights[i]} * ${
                            outputs[i]
                        }`
                    );

                    weightedSum += neuron.weights[i] * outputs[i];
                }
                console.log(`FWD-PROP_ weightedSum: ${weightedSum}`);

                const activation: number = this.sigmoid(
                    weightedSum + neuron.bias
                );
                newOutputs.push(activation);
            }
            outputs = newOutputs;
        }
        console.log(`FWD-PROP_ output:`, outputs);

        return outputs;
    }

    /**
     * Gets the highest result from the forward propagation algorithm
     */
    public getResult(results: number[]): number {
        let resultList: number[] = [];
        for (let i = 0; i < results.length; i++) {
            resultList.push(Math.round(results[i]));
        }
        const result: number = Math.max(...resultList);

        return result;
    }

    // Sigmoid functions
    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    private sigmoidDerivative(x: number): number {
        return this.sigmoid(x) * (1 - this.sigmoid(x));
    }

    // Training algorithm
    public train(
        inputs: number[],
        targets: number[],
        learnRate: number = 0.1
    ): void {
        // Get results from forward propagation algorithm
        const outputs: number[] = this.forwardPropagation(inputs);

        // Error calculation per neuron
        console.log(`TRAINING_ fwdProp-outputs:`, outputs);
        const outputErrors: number[] = [];
        for (let i = 0; i < outputs.length; i++) {
            console.log(
                `TRAINING_ err-calc (${i + 1}/${outputs.length}) target: ${
                    targets[i]
                } network: ${outputs[i]} error: ${targets[i] - outputs[i]}`
            );

            outputErrors.push(targets[i] - outputs[i]);
        }
        let networkError: number = 0;
        for (let j = 0; j < outputErrors.length; j++) {
            networkError += outputErrors[j];
        }
        networkError = networkError / outputErrors.length;
        console.log(`TRAINING_ networkError: ${networkError}`);

        // Back propagation algorithm for training neural networks
        // Output layer training
        for (let i = 0; i < this.layers[this.layers.length - 1].length; i++) {
            const neuron: Neuron = this.layers[this.layers.length - 1][i];

            console.log(
                `TRAINING_ OUTPUT-LAYER neuron ${i + 1}/${
                    this.layers[this.layers.length - 1].length
                }`
            );
            const oldBias = neuron.bias;
            neuron.bias +=
                outputErrors[i] *
                this.sigmoidDerivative(outputs[i]) *
                learnRate;
            console.log(
                `TRAINING_ OUTPUT-LAYER bias: ${oldBias} => ${neuron.bias}`
            );
            // weights
            for (let j = 0; j < neuron.weights.length; j++) {
                const oldWeight = neuron.weights[j];
                neuron.weights[j] +=
                    outputErrors[i] *
                    this.sigmoidDerivative(outputs[i]) *
                    inputs[j] *
                    learnRate;
                console.log(
                    `TRAINING_ OUTPUT-LAYER weight (${j + 1}/${
                        neuron.weights.length
                    }): ${oldWeight} => ${neuron.weights[j]}`
                );
            }
        }

        // Hidden layer(s) training
    }
}

//       for (let j = 0; j < neuron.weights.length; j++) {
//         if (outputErrors[i] == undefined)
//           console.log(`TRAINING_ missing outputErrors[j:${j}]`);
//         console.log(
//           `TRAINING_ weight: j:${neuron.weights[j]} outputErrors: j:${outputErrors[j]} inputs: i:${inputs[j]} LR: ${learningRate}`
//         );
//         neuron.weights[j] +=
//           outputErrors[i] *
//           this.sigmoidDerivative(outputs[i]) *
//           inputs[j] *
//           learningRate;
//       }
//     }
//     // update biases and weights for hidden layers
//     for (let i = this.layers.length - 2; i > 0; i--) {
//       for (const neuron of this.layers[i]) {
//         let error: number = 0;
//         for (let j = 0; j < this.layers[i + 1].length; j++) {
//           console.log(
//             `TRAINING_ bias: ${neuron.bias} outputErrors: i:${outputErrors[i]} LR: ${learningRate}`
//           );
//           console.log(
//             `TRAINING_ weight: j:${neuron.weights[j]} outputErrors: i:${outputErrors[i]} inputs: j:${inputs[j]} LR: ${learningRate}`
//           );
//           error +=
//             this.layers[i + 1][j].weights[this.layers[i].indexOf(neuron)] *
//             outputErrors[j];
//         }
//         let weightedSum = 0;
//         for (let k = 0; k < neuron.weights.length; k++) {
//           weightedSum += neuron.weights[k] * inputs[k];
//         }

//         error *= this.sigmoidDerivative(weightedSum + neuron.bias);

//         neuron.bias += error * learningRate;
//         for (let j = 0; j < neuron.weights.length; j++) {
//           neuron.weights[j] += error * inputs[j] * learningRate;
//         }
//       }
//     }
//   }
