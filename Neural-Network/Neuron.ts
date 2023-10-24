import { Random } from '../Util/math.ts';
export class Neuron {
    public weights: number[];
    public bias: number;

    /**
     *
     * @param inputs The number of inputs for this neuron
     */
    constructor(inputs: number) {
        this.weights = Array.from({ length: inputs }, () => Random(-1, 1));

        this.bias = Random(-1, 1);
    }

    /**
     *
     * @param inputs The inputs of this neuron to calculate the output of this neuron
     * @returns The weighted sum of the neuron after being passed through the activation function
     */
    public calculateOutput(inputs: number[]): number {
        if (inputs.length !== this.weights.length)
            throw new Error('Number of inputs must match number of weights');

        const weightedSum =
            inputs.reduce(
                (sum, input, index) => sum + input * this.weights[index],
                0
            ) + this.bias;
        return this.sigmoid(weightedSum);
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
     * @param inputs The inputs used to train the layer the neuron is in
     * @param error The error of the neuron
     * @param learnRate The rate at which the network learns
     */
    public updateWeights(
        inputs: number[],
        error: number,
        learnRate: number
    ): void {
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += learnRate * error * inputs[i];
        }
    }

    /**
     *
     * @param error The error of the neuron
     * @param learnRate The rate at which the network learns
     */
    public updateBias(error: number, learnRate: number): void {
        this.bias += learnRate * error;
    }
}
