import { Neuron } from "./Neuron";

export class NeuralNetwork {
	public layers: Neuron[][];

	constructor(layerSizes: number[]) {
		this.layers = [];

		console.log(`INIT_ layerSizes: ${layerSizes}`);
		console.log(`INIT_ layers: ${layerSizes.length}`);

		// Input layer
		let inputLayerNeurons: Neuron[] = [];
		for (let i = 0; i < layerSizes[0]; i++) {
			console.log(
				`INIT-IN_ new neuron: ${i + 1}/${layerSizes[0]} inputs: 1`
			);

			inputLayerNeurons.push(new Neuron(1));
		}
		this.layers.push(inputLayerNeurons);

		// Hidden layer(s)
		for (let i = 1; i < layerSizes.length - 1; i++) {
			let layerNeurons: Neuron[] = [];
			for (let j = 0; j < layerSizes[i]; j++) {
				console.log(
					`INIT-HL_ layer: ${i}/${
						layerSizes.length - 2
					} new neuron: ${j + 1}/${layerSizes[i]} inputs: ${
						layerSizes[i - 1]
					}`
				);
				layerNeurons.push(new Neuron(layerSizes[i - 1]));
			}
			this.layers.push(layerNeurons);
		}

		// Output layer
		let outputLayerNeurons: Neuron[] = [];
		for (let i = 0; i < layerSizes[layerSizes.length - 1]; i++) {
			console.log(
				`INIT-OUT_ new neuron: ${i + 1}/${
					layerSizes[layerSizes.length - 1]
				} inputs: ${layerSizes.length - 2}`
			);
			outputLayerNeurons.push(new Neuron(layerSizes.length - 2));
		}
		this.layers.push(outputLayerNeurons);
	}

	/**
	 * Forward Propagation Algorithm
	 */
	public forwardPropagation(inputs: number[]): number[] {
		let outputs: number[] = inputs;

		for (const layer of this.layers) {
			const newOutputs: number[] = [];
			for (const neuron of layer) {
				console.log(
					`FWD-PROP_ weights: ${neuron.weights} bias: ${neuron.bias}`
				);
				let weightedSum: number = 0;
				for (let i = 0; i < neuron.weights.length; i++) {
					console.log(
						`FWD-PROP_ weightedSum-equation: (${i + 1}/${
							neuron.weights.length
						}) ${weightedSum} + ${neuron.weights[i]} * ${
							outputs[i]
						}`
					);

					weightedSum += neuron.weights[i] * outputs[i];
				}
				console.log(`FWD-PROP_ weightedSum: ${weightedSum}`);

				const activation: number = this.sigmoid(
					weightedSum + neuron.bias
				);
				newOutputs.push(activation);
			}
			outputs = newOutputs;
		}
		console.log(`FWD-PROP_ output:`, outputs);

		return outputs;
	}

	/**
	 * Gets the highest result from the forward propagation algorithm
	 */
	public getResult(results: number[]): number {
		let resultList: number[] = [];
		for (let i = 0; i < results.length; i++) {
			resultList.push(Math.round(results[i]));
		}
		const result: number = Math.max(...resultList);

		return result;
	}

	// Sigmoid functions
	private sigmoid(x: number): number {
		return 1 / (1 + Math.exp(-x));
	}

	private sigmoidDerivative(x: number): number {
		return this.sigmoid(x) * (1 - this.sigmoid(x));
	}

	// Training algorithm
	public train(
		inputs: number[],
		targets: number[],
		learnRate: number = 0.1
	): void {
		// Get results from forward propagation algorithm
		const outputs: number[] = this.forwardPropagation(inputs);
		console.log(`TRAINING_ learnrate: ${learnRate}`);

		// Error calculation per neuron
		console.log(`TRAINING_ fwdProp-outputs:`, outputs);
		const outputErrors: number[] = [];
		for (let i = 0; i < outputs.length; i++) {
			console.log(
				`TRAINING_ err-calc (${i + 1}/${outputs.length}) target: ${
					targets[i]
				} network: ${outputs[i]} error: ${targets[i] - outputs[i]}`
			);

			outputErrors.push(targets[i] - outputs[i]);
		}
		let networkError: number = 0;
		for (let j = 0; j < outputErrors.length; j++) {
			networkError += outputErrors[j];
		}
		networkError = networkError / outputErrors.length;
		console.log(`TRAINING_ networkError: ${networkError}`);

		// Back propagation algorithm for training neural networks
		// Output layer training
		for (let i = 0; i < this.layers[this.layers.length - 1].length; i++) {
			// loop through each neuron
			const neuron: Neuron = this.layers[this.layers.length - 1][i];

			console.log(
				`TRAINING_ OUTPUT-LAYER neuron ${i + 1}/${
					this.layers[this.layers.length - 1].length
				}`
			);
			const oldBias = neuron.bias;
			neuron.bias +=
				outputErrors[i] *
				this.sigmoidDerivative(outputs[i]) *
				learnRate;
			console.log(
				`TRAINING_ OUTPUT-LAYER bias: ${oldBias} => ${neuron.bias}`
			);
			// weights
			for (let j = 0; j < neuron.weights.length; j++) {
				// loop through each neurons' weights
				const oldWeight = neuron.weights[j];
				neuron.weights[j] +=
					outputErrors[i] *
					this.sigmoidDerivative(outputs[i]) *
					inputs[j] *
					learnRate;
				console.log(
					`TRAINING_ OUTPUT-LAYER weight (${j + 1}/${
						neuron.weights.length
					}): ${oldWeight} => ${neuron.weights[j]}`
				);
			}
		}

		// Hidden layer(s) training
		for (let i = this.layers.length - 2; i > 0; i--) {
			// loop through each hidden layer
			for (let n = 0; n < this.layers[i].length; n++) {
				// loop through each neuron
				const neuron: Neuron = this.layers[i][n];
				let error: number = 0;
				for (let j = 0; j < this.layers[i + 1].length; j++) {
					// calculate errors for weights
					console.log(
						`TRAINING_ HIDDEN-LAYER layer ${i}/${
							this.layers.length - 2
						} neuron ${n + 1}/${this.layers[i].length}`
					);
					error +=
						this.layers[i + 1][j].weights[
							this.layers[i].indexOf(neuron)
						] * outputErrors[j];
				}
				let weightedSum: number = 0;
				for (let k = 0; k < neuron.weights.length; k++) {
					// loop through weights and calculate weightedSum to calculate new error
					weightedSum += neuron.weights[k] * inputs[k];
				}

				error *= this.sigmoidDerivative(weightedSum + neuron.bias);

				const oldBias = neuron.bias;
				neuron.bias += error * learnRate;
				console.log(
					`TRAINING_ HIDDEN-LAYER ${i}/${
						this.layers.length - 2
					} bias: ${oldBias} => ${neuron.bias}`
				);

				for (let L = 0; L < neuron.weights.length; L++) {
					// loop through weights and update them
					const oldWeight = neuron.weights[L];
					neuron.weights[L] += error * inputs[L] * learnRate;
					console.log(
						`TRAINING_ HIDDEN-LAYER ${i}/${
							this.layers.length - 2
						} weight (${L + 1}/${
							neuron.weights.length
						}): ${oldWeight} => ${neuron.weights[L]}`
					);
				}
			}
		}
	}
}
