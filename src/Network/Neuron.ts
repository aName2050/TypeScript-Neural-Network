import { Calculus } from "../Math/Calculus";
import { Random } from "../Math/Math";
import { Activation } from "../Math/Activation";

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
}
