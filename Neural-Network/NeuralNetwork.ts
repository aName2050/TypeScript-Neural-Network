import { Layer } from "./Layer";
import { Neuron } from "./Neuron";

export class NeuralNetwork {
	public layers: Layer[];

	/**
	 *
	 * @param layerSizes The sizes of each layer
	 * @example new NeuralNetwork([2, 1, 1]); // 2 neurons for input, 1 neuron for 1 hidden layer, 1 neuron for output
	 */
	constructor(layerSizes: number[]) {
		this.layers = [];

		console.log(`INIT_ new layer (INPUT) neurons: ${layerSizes[0]}`);
		this.layers.push(new Layer(layerSizes[0], 0, true));
		// Hidden layer(s) and output layer initialization
		for (let i = 1; i < layerSizes.length; i++) {
			console.log(
				`INIT_ new layer  (${i + 1}/${layerSizes.length}) neurons: ${
					layerSizes[i]
				}`
			);
			const neuronsInPrevLayer: number = layerSizes[i - 1];
			const neuronsInCurrLayer: number = layerSizes[i];
			this.layers.push(new Layer(neuronsInCurrLayer, neuronsInPrevLayer));
		}
	}

	/**
	 *
	 * @param inputs The inputs to pass into the network
	 * @returns The processed output
	 */
	public forwardPropagation(inputs: number[]): number[] {
		let outputs: number[] = [];
		for (let i = 0; i < this.layers.length; i++) {
			outputs = this.layers[i].calculateLayerOutputs(inputs);
		}
		return outputs;
	}

	/**
	 * Custom training algoritm
	 * @param inputs The inputs to the network
	 * @param targets The target outputs of the provided inputs
	 * @param learnRate The rate at which the network "learns"
	 */
	public train(
		inputs: number[],
		targets: number[],
		learnRate: number = 0.1
	): void {
		// Get network output
		const outputs: number[] = this.forwardPropagation(inputs);
		// Error calculation
		let networkError: number = 0;
		let outputErrors: number[] = [];
		for (let i = 0; i < outputs.length; i++) {
			networkError += outputs[i] - targets[i];
			outputErrors.push(outputs[i] - targets[i]);
		}
		console.log(
			`TRAINING_ network error: (+) ${Math.abs(
				networkError
			)} network output: ${outputs}`
		);
		// Output layer
		for (
			let i = 0;
			i < this.layers[this.layers.length - 1].Neurons.length;
			i++
		) {
			const neuron: Neuron =
				this.layers[this.layers.length - 1].Neurons[i];

			// Update bias
			const oldBias: number = neuron.bias;
			neuron.bias +=
				outputErrors[i] *
				this.sigmoidDerivative(outputs[i]) *
				learnRate;
			console.log(
				`TRAINING_ OUTPUT-LAYER neuron (${i + 1}/${
					this.layers[this.layers.length - 1].Neurons.length
				}) bias: ${oldBias} => ${neuron.bias}`
			);

			// Update weights
			for (let j = 0; j < neuron.weights.length; j++) {
				const oldWeight: number = neuron.weights[j];
				neuron.weights[j] +=
					outputErrors[i] *
					this.sigmoidDerivative(outputs[i]) *
					inputs[j] *
					learnRate;
				console.log(
					`TRAINING_ OUTPUT-LAYER neuron (${i + 1}/${
						this.layers[this.layers.length - 1].Neurons.length
					}) weight ${j + 1}/${
						neuron.weights.length
					}: ${oldWeight} => ${neuron.weights[j]}`
				);
			}
		}

		// Hidden layers
		// loop through each hidden layer
		for (let i = 1; i < this.layers.length - 1; i++) {
			// loop through each neuron
			for (let n = 0; n < this.layers[i].Neurons.length; n++) {
				const neuron: Neuron = this.layers[i].Neurons[n];
				let error: number = 0;
				// calculate errors for each neuron
				for (let j = 0; j < this.layers[i + 1].Neurons.length; j++) {
					error +=
						this.layers[i + 1].Neurons[j].weights[n] *
						outputErrors[j];
				}
				error *= this.sigmoidDerivative(
					neuron.weights.reduce(
						(sum, weight, index) => sum + weight * inputs[index]
					) + neuron.bias
				);

				// update neuron weights and bias
				const oldBias = neuron.bias;
				neuron.bias += error * learnRate;
				console.log(
					`TRAINING_ HIDDEN-LAYER neuron (${n + 1}/${
						this.layers[this.layers.length - 1].Neurons.length
					}) bias: ${oldBias} => ${neuron.bias}`
				);
				for (let j = 0; j < neuron.weights.length; j++) {
					const oldWeight = neuron.weights[j];
					neuron.weights[j] += error * inputs[j] * learnRate;
					console.log(
						`TRAINING_ HIDDEN-LAYER neuron (${n + 1}/${
							this.layers[this.layers.length - 1].Neurons.length
						}) weight ${j + 1}/${
							neuron.weights.length
						}: ${oldWeight} => ${neuron.weights[j]}`
					);
				}
			}
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
