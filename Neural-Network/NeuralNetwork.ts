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

        console.log(`INIT_ new layer (INPUT) neurons: ${layerSizes[0]}`);
        this.layers.push(new Layer(layerSizes[0], 0, true));
        // Hidden layer(s) and output layer initialization
        for (let i = 1; i < layerSizes.length; i++) {
            console.log(
                `INIT_ new layer  (${i + 1}/${layerSizes.length}) neurons: ${
                    layerSizes[i]
                }`
            );
            const neuronsInPrevLayer: number = layerSizes[i - 1];
            const neuronsInCurrLayer: number = layerSizes[i];
            this.layers.push(new Layer(neuronsInCurrLayer, neuronsInPrevLayer));
        }
    }

    /**
     *
     * @param inputs The inputs to pass into the network
     * @returns The processed output
     */
    public forwardPropagation(inputs: number[]): number[] {
        // coming soon!
        return [NaN];
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
        // coming soon!
    }
}
