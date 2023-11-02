import { Functions } from '../Math/Functions';
import { Random } from '../Math/Math';
import { Activation } from '../Math/Activation';

const functions = new Functions();
const activation = new Activation();

export class Neuron {
	public bias: number;
	public weights: number[];

	/**
	 *
	 * @param inputs The number of inputs going into the neuron
	 * @param inputNode Determine whether the current neuron is in the first layer
	 */
	constructor(inputs: number, inputNode?: boolean) {
		if (inputNode) {
			this.bias = 0;
			this.weights = [0];
		}

		this.bias = Random(1, -1);
		this.weights = Array.from({ length: inputs }, () => Random(-1, 1));
	}

	/**
	 *
	 * @param neuronInputs The inputs of the neuron
	 * @returns The output of the neuron
	 */
	public CalculateNeuronActivation(neuronInputs: number[]): number {
		let z: number = functions.WeightedSumCalculation(
			neuronInputs,
			this.weights
		);
		return activation.Sigmoid(z + this.bias);
	}
}
