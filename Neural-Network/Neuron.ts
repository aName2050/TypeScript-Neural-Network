import { Random } from "../Util/math.ts";
export class Neuron {
	public weights: number[];
	public bias: number;

	/**
	 *
	 * @param inputs The number of inputs for this neuron
	 */
	constructor(inputs: number, noWeightsAndBiases: boolean = false) {
		if (noWeightsAndBiases) {
			this.weights = [1];
			this.bias = 0;
		} else {
			this.weights = Array.from({ length: inputs }, () => Random(-1, 1));

			this.bias = Random(-1, 1);
		}
	}

	/**
	 *
	 * @param inputs The inputs of the neuron
	 * @returns The output of the neuron
	 */
	public calculateNeuronOutput(inputs: number[]): number {
		let output: number = 0;
		const weightedSum: number = inputs.reduce(
			(sum, weight, index) => sum + inputs[index] * weight
		);
		output = this.sigmoid(weightedSum + this.bias);
		return output;
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

	/**
	 *
	 * @param x The number to pass to the Sigmoid function
	 * @returns The number squished between 0 and 1
	 */
	private sigmoid(x: number): number {
		return 1 / (1 + Math.exp(-x));
	}
}
