import { Layer } from './Layer';
import { Neuron } from './Neuron';

export class NeuralNetwork {
    public layers: Layer[];

    /**
     *
     * @param layerSizes The sizes of each layer
     * @example new NeuralNetwork([2, 1, 1]); // 2 neurons for input, 1 neuron for 1 hidden layer, 1 neuron for output
     */
    constructor(layerSizes: number[]) {
        this.layers = [];

        for (let i = 1; i < layerSizes.length; i++) {
            const neuronsInPrevLayer = layerSizes[i - 1];
            const neuronsInCurrLayer = layerSizes[i];
            console.log(
                `INIT_ new layer: ${layerSizes[i]} neurons w/ ${
                    layerSizes[i - 1]
                } inputs`
            );
            this.layers.push(new Layer(neuronsInCurrLayer, neuronsInPrevLayer));
        }
    }

    /**
     *
     * @param inputs The inputs to pass into the network
     * @returns The processed output
     */
    public forwardPropagation(inputs: number[]): number[] {
        let outputs = inputs;
        for (const layer of this.layers) {
            outputs = layer.getOutputs(outputs);
        }
        return outputs;
    }

    /**
     * Custom training algoritm
     * @param inputs The inputs to the network
     * @param targets The target outputs of the provided inputs
     * @param learnRate The rate at which the network "learns"
     */
    public train(
        inputs: number[],
        targets: number[],
        learnRate: number = 0.1
    ): void {
        let layerInputs = inputs;

        for (let i = 0; i < this.layers.length; i++) {
            console.log(`TRAINING_ ${i + 1}/${this.layers.length} layers`);
            const layer = this.layers[i];
            const layerTargets = targets.slice(i, i + 1); // Seperate out the targets for the current layer
            layer.trainLayer(layerInputs, layerTargets, learnRate);
            layerInputs = layer.getOutputs(layerInputs); // Prepare inputs for next layer training
        }
    }
}
