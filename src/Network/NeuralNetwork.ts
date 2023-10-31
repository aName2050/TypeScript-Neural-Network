import { Neuron } from "./Neuron";

export class NeuralNetwork {
	public Neurons: Neuron[][];

	/**
	 *
	 * @param layerSizes The sizes of each layer
	 * The first parameter is always the input layer and the last parameter is always the output layer
	 * @example
	 * const NN = new NeuralNetwork(2, 1, 1);
	 */
	constructor(...layerSizes: number[]) {
		this.Neurons = Array.from({ length: layerSizes.length }, () => []);

		// Initialize input layer
		for (let i = 0; i < layerSizes[0]; i++) {
			this.Neurons[0].push(new Neuron(0, true));
			console.log(
				`INIT_ INPUT-LAYER new neuron (${i + 1}/${layerSizes[0]})`
			);
		}
		// Initialize hidden layer(s)
		for (let i = 1; i < layerSizes.length - 1; i++) {
			for (let j = 0; j < layerSizes[i]; j++) {
				this.Neurons[i].push(new Neuron(layerSizes[i - 1], false));
				console.log(
					`INIT_ HIDDEN-LAYER (${i}/${
						layerSizes.length - 2
					}) new neuron (${j + 1}/${layerSizes[i]})`
				);
			}
		}
		// Initialize output layer
		for (let i = 0; i < layerSizes[layerSizes.length - 1]; i++) {
			this.Neurons[this.Neurons.length - 1].push(
				new Neuron(layerSizes[layerSizes.length - 1], false)
			);
			console.log(
				`INIT_ OUTPUT-LAYER new neuron (${i + 1}/${
					layerSizes[layerSizes.length - 1]
				})`
			);
		}
	}
}
