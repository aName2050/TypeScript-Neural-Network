import { Neuron } from "./Neuron.ts";

export class Layer {
	public Neurons: Neuron[];

	/**
	 *
	 * @param neurons The number of neurons in this layer
	 * @param inputs The number of inputs for each neuron
	 */
	constructor(neurons: number, inputs: number, inputLayer: boolean = false) {
		this.Neurons = [];

		for (let i = 0; i < neurons; i++) {
			console.log(
				`INIT_ new neuron (${
					i + 1
				}/${neurons}) inputs: ${inputs} inputLayerNeuron?: ${inputLayer}`
			);

			this.Neurons.push(new Neuron(inputs, inputLayer));
		}
	}

	/**
	 *
	 * @param inputs The inputs to the layer
	 * @returns The layer's output
	 */
	public calculateLayerOutputs(inputs: number[]): number[] {
		let output: number[] = [];
		for (let i = 0; i < this.Neurons.length; i++) {
			const neuron = this.Neurons[i];
			output.push(neuron.calculateNeuronOutput(inputs));
		}
		return output;
	}
}
