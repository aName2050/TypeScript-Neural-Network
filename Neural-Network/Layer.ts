import { Neuron } from './Neuron.ts';

export class Layer {
    public Neurons: Neuron[];

    /**
     *
     * @param neurons The number of neurons in this layer
     * @param inputs The number of inputs for each neuron
     */
    constructor(neurons: number, inputs: number) {
        this.Neurons = [];

        for (let i = 0; i < neurons; i++) {
            this.Neurons.push(new Neuron(inputs));
        }
    }

    /**
     *
     * @param inputs The inputs to the neurons in this layer
     * @returns The outputs of this layer
     */
    public getOutputs(inputs: number[]): number[] {
        const outputs: number[] = [];
        for (const neuron of this.Neurons) {
            outputs.push(neuron.calculateOutput(inputs));
        }
        return outputs;
    }

    /**
     *
     * @param inputs The inputs of the layer to train with
     * @param targets The target outputs of the provided inputs
     * @param learnRate The rate at which the network learns
     */
    public trainLayer(
        inputs: number[],
        targets: number[],
        learnRate: number
    ): void {
        for (let i = 0; i < this.Neurons.length; i++) {
            const neuron = this.Neurons[i];
            const output = neuron.calculateOutput(inputs);
            const error = targets[i] - output;
            neuron.updateWeights(inputs, error, learnRate);
        }
    }
}
