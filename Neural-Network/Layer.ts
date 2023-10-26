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

    /**
     *
     * @param x The number to pass to the Sigmoid function
     * @returns The number squished between 0 and 1
     */
    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     *
     * @param x The number to pass to the sigmoid derivative function
     * @returns The result of the function
     */
    private sigmoidDerivative(x: number): number {
        const sigmoid: number = this.sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }
}
