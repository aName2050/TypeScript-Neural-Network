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
            console.log(
                `INIT_ new neuron (${i + 1}/${neurons}) inputs: ${inputs}`
            );
            this.Neurons.push(new Neuron(inputs));
        }
    }
}
