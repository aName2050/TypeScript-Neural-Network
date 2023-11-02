import { Functions } from '../Math/Functions';
import { Neuron } from './Neuron';

export class NeuralNetwork {
	public Neurons: Neuron[][];

	/**
	 *
	 * @param layerSizes The sizes of each layer
	 * The first parameter is always the input layer and the last parameter is always the output layer
	 * The hidden layers must be smaller then the previous layer to avoid `NaN` outputs
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
					`INIT_ HIDDEN-LAYER (${i}/${layerSizes.length - 2
					}) new neuron (${j + 1}/${layerSizes[i]}) inputs ${layerSizes[i - 1]
					}`
				);
			}
		}
		// Initialize output layer
		for (let i = 0; i < layerSizes[layerSizes.length - 1]; i++) {
			this.Neurons[this.Neurons.length - 1].push(
				new Neuron(layerSizes[layerSizes.length - 1], false)
			);
			console.log(
				`INIT_ OUTPUT-LAYER new neuron (${i + 1}/${layerSizes[layerSizes.length - 1]
				}) inputs ${layerSizes[layerSizes.length - 2]}`
			);
		}
	}

	/**
	 *
	 * @param inputs The inputs of the network
	 * @returns THe output of the network
	 */
	public forwardPropagation(inputs: number[]): number[] {
		let outputs: number[] = [];
		// Loop through each layer, skipping the input layer
		let layerOutput: number[] = inputs;
		for (let i = 1; i < this.Neurons.length; i++) {
			let tmp: number[] = [];
			for (let j = 0; j < this.Neurons[i].length; j++) {
				console.log(
					`FWD-PROP_ layer (${i}/${this.Neurons.length - 1
					}) neuron (${j + 1}/${this.Neurons[i].length
					}) > prev layerOutput: `,
					layerOutput
				);

				tmp.push(
					this.Neurons[i][j].CalculateNeuronActivation(layerOutput)
				);
			}
			console.log(
				`FWD-PROP_ layer (${i}/${this.Neurons.length - 1
				}) > curr layerOutput: `,
				tmp
			);
			layerOutput = tmp;
			outputs = layerOutput;
		}
		console.log(`FWD-PROP_ networkOutput: `, outputs);
		return outputs;
	}

	public train(inputs: number[], targets: number[], learnRate: number): void {
		const networkOutput: number[] = this.forwardPropagation(inputs);

		const network: Functions = new Functions();
		const cost: number = network.Cost(networkOutput, targets);

		// !! CALCULUS !!
		/** Chain Rule
		 * !! PD = Partial Derivative !!
		 * !! PO = Predicted Output !!
		 * !! NO = Network Output !!
		 * !! C = Cost function !!
		 * !! z = weightedSum? !!
		 * !! S = Sigmoid function !!
		 * !! x = Network Input !!
		 * !! w = Weight !!
		 * !! b =  Bias !!
		 * 
		 * PD of C / PD of w[i]
		 * =
		 * PD of C / PD of PO
		 * x
		 * PD of PO / PD of z
		 * x
		 * PD of z / PD of w[i]
		 */
		/** Gradients
		 * Gradient of the Cost function with respect to the predicted value
		 * PD of C / PD of PO = 2/n * SUM(NO[i] - PO[i])
		 * 
		 * Gradient of the predicted value wiith respect to z
		 * PD of PO / PD of z = S(z) * (1 - S(z))
		 * 
		 * Gradient of z with respect to w[i]
		 * x[i]
		 * 
		 * Weights
		 * PD of C / PD of w[i] = 2/n * SUM(NO[i] - PO[i]) * S(z) * (1 - S(z)) * x[i]
		 * Biases
		 * PD of C / PD of b = 2/n * SUM(NO[i] - PO[i]) * S(z) * (1 - S(z))
		 */
		/** Optimization of the network
		 * Gradient Descent
		 * 
		 */
	}
}
