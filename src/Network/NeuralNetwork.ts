import { Functions } from '../Math/Functions';
import { Neuron } from './Neuron';
import { GradientDescent } from '../Math/Gradients';

import chalk from 'chalk';

export class NeuralNetwork {
    public Neurons: Neuron[][];

    /**
     *
     * @param layerSizes The sizes of each layer
     * The first parameter is always the input layer and the last parameter is always the output layer
     * The hidden layers must be smaller then the previous layer to avoid `NaN` outputs
     * @example
     * const NN = new NeuralNetwork(2, 1, 1);
     */
    constructor(...layerSizes: number[]) {
        this.Neurons = Array.from({ length: layerSizes.length }, () => []);

        // Initialize input layer
        for (let i = 0; i < layerSizes[0]; i++) {
            this.Neurons[0].push(new Neuron(0, true));
            console.log(
                `${chalk.white.bgGreen.bold('INIT_')} INPUT-LAYER new neuron (${
                    i + 1
                }/${layerSizes[0]})`
            );
        }
        // Initialize hidden layer(s)
        for (let i = 1; i < layerSizes.length - 1; i++) {
            for (let j = 0; j < layerSizes[i]; j++) {
                this.Neurons[i].push(new Neuron(layerSizes[i - 1], false));
                console.log(
                    `${chalk.white.bgGreen.bold('INIT_')} HIDDEN-LAYER (${i}/${
                        layerSizes.length - 2
                    }) new neuron (${j + 1}/${layerSizes[i]}) inputs ${
                        layerSizes[i - 1]
                    }`
                );
            }
        }
        // Initialize output layer
        for (let i = 0; i < layerSizes[layerSizes.length - 1]; i++) {
            this.Neurons[this.Neurons.length - 1].push(
                new Neuron(layerSizes[layerSizes.length - 1], false)
            );
            console.log(
                `${chalk.white.bgGreen.bold(
                    'INIT_'
                )} OUTPUT-LAYER new neuron (${i + 1}/${
                    layerSizes[layerSizes.length - 1]
                }) inputs ${layerSizes[layerSizes.length - 2]}`
            );
        }
    }

    /**
     *
     * @param inputs The inputs of the network
     * @returns THe output of the network
     */
    public forwardPropagation(inputs: number[]): number[] {
        let outputs: number[] = [];
        // Loop through each layer, skipping the input layer
        let layerOutput: number[] = inputs;
        for (let i = 1; i < this.Neurons.length; i++) {
            let tmp: number[] = [];
            for (let j = 0; j < this.Neurons[i].length; j++) {
                console.log(
                    `${chalk.white.bgMagenta.bold('FWD-PROP_')} layer (${i}/${
                        this.Neurons.length - 1
                    }) neuron (${j + 1}/${
                        this.Neurons[i].length
                    }) > prev layerOutput: `,
                    layerOutput
                );

                tmp.push(
                    this.Neurons[i][j].CalculateNeuronActivation(layerOutput)
                );
            }
            console.log(
                `${chalk.white.bgMagenta.bold('FWD-PROP_')} layer (${i}/${
                    this.Neurons.length - 1
                }) > curr layerOutput: `,
                tmp
            );
            layerOutput = tmp;
            outputs = layerOutput;
        }
        console.log(
            `${chalk.white.bgMagenta.bold('FWD-PROP_')} networkOutput: `,
            outputs
        );
        return outputs;
    }

    public train(inputs: number[], targets: number[], learnRate: number): void {
        const networkOutput: number[] = this.forwardPropagation(inputs);

        const network: Functions = new Functions();
        network.Cost(networkOutput, targets);

        // Neural networks are just a series of mathematical calculations happening in a specific and correct order
        // And life is just a series of chemical reactions happening in a specific and correct order
        // Working on neural networks really gets you thinking we're in a simulation :D
        // !! CALCULUS !!
        /*
         * !! PD = Partial Derivative !!
         * !! PO = Predicted Output !!
         * !! NO = Network Output !!
         * !! C = Cost function !!
         * !! z = weightedSum? !!
         * !! S = Sigmoid function !!
         * !! x = Network Input !!
         * !! w = Weight !!
         * !! b =  Bias !!
         * !! LR = Learn Rate !!
         */
        /* Chain Rule
         * PD of C / PD of w[i]
         * =
         * PD of C / PD of PO
         * x
         * PD of PO / PD of z
         * x
         * PD of z / PD of w[i]
         */
        /*
         * Gradient of the Cost function with respect to the predicted value
         * PD of C / PD of PO = 2/n * SUM(NO[i] - PO[i])
         *
         * Gradient of the predicted value wiith respect to z
         * PD of PO / PD of z = S(z) * (1 - S(z))
         *
         * Gradient of z with respect to w[i]
         * x[i]
         */
        // PD of C / PD of w[i] = 2/n * SUM(NO[i] - PO[i]) * S(z) * (1 - S(z)) * x[i]
        // PD of C / PD of b = 2/n * SUM(NO[i] - PO[i]) * S(z) * (1 - S(z))
        // w[i] = w[i] - (LR * GW)
        // b = b - (LR * GB)

        // Backpropagation & Gradient Descent algorithms
        const gradientDescent: GradientDescent = new GradientDescent();
        // Backpropagation: going through the network backwards to train it
        let layerOutput: number[] = inputs;
        for (let layer = 1; layer < this.Neurons.length; layer++) {
            // Iterate over each neuron in the current layer
            let activeLayerOutputTemp: number[] = [];
            for (
                let neuron = 0;
                neuron < this.Neurons[layer].length;
                neuron++
            ) {
                console.log(
                    `${chalk.black.bgWhite.bold(
                        'TRAINING_'
                    )} backProp layer (${layer}/${
                        this.Neurons.length
                    }) neuron (${neuron + 1}/${this.Neurons[layer].length})`
                );
                console.log(
                    `${chalk.white.bgYellow.bold(
                        'TRAINING_'
                    )} backProp neuron data:`,
                    this.Neurons[layer][neuron]
                );
                // Calculate weightedSum for gradient calculations
                console.log(
                    `${chalk.white.bgBlue.bold(
                        'TRAINING_'
                    )} z > inputs:${inputs} weights:${
                        this.Neurons[layer][neuron].weights
                    }`
                );
                const z: number =
                    this.Neurons[layer][neuron].CalculateNeuronActivation(
                        layerOutput
                    );
                activeLayerOutputTemp.push(z);
                layerOutput = activeLayerOutputTemp;
                console.log(`${chalk.white.bgBlue.bold('TRAINING_')} z = ${z}`);
                // // Calculate gradients for gradient descent algorithm
                const dC_dWi: number = gradientDescent.partialCostPartialWi(
                    networkOutput[neuron],
                    targets[neuron],
                    z,
                    inputs[neuron]
                );
                const dC_dB: number = gradientDescent.partialCostPartialB(
                    networkOutput[neuron],
                    targets[neuron],
                    z
                );
                console.log(
                    `${chalk.white.bgBlue.bold(
                        'TRAINING_'
                    )} dC_dWi: ${dC_dWi} dC_dB: ${dC_dB}`
                );
                // // Update weights
                for (
                    let i = 0;
                    i < this.Neurons[layer][neuron].weights.length;
                    i++
                ) {
                    const oldWeight: number =
                        this.Neurons[layer][neuron].weights[i];
                    this.Neurons[layer][neuron].weights[i] =
                        gradientDescent.UpdateWeight(
                            this.Neurons[layer][neuron].weights[i],
                            learnRate,
                            dC_dWi
                        );
                    console.log(
                        `${chalk.white.bgGreenBright.bold(
                            'TRAINING_'
                        )} updated weight: ${oldWeight} => ${
                            this.Neurons[layer][neuron].weights[i]
                        }`
                    );
                }
                // Update bias
                const oldBias: number = this.Neurons[layer][neuron].bias;
                this.Neurons[layer][neuron].bias = gradientDescent.UpdateBias(
                    this.Neurons[layer][neuron].bias,
                    learnRate,
                    dC_dB
                );
                console.log(
                    `${chalk.white.bgGreenBright.bold(
                        'TRAINING_'
                    )} updated bias: ${oldBias} => ${
                        this.Neurons[layer][neuron].bias
                    }`
                );
            }
        }
    }
}
