import { Activation } from "../Math/Activation";
import { Neuron } from "../Network/Neuron";
const sigmoid = new Activation().Sigmoid;

// Single neuron gradient calculation
class Gradients {
	// PD of C / PD of w[i] = 2/n * SUM(NO[i] - PO[i]) * S(z) * (1 - S(z)) * x[i]
	public Weights(
		networkOutputs: number[],
		weightedSum: number,
		networkInput: number,
		targetOutputs: number[],
		networkInputsLength: number
	): number {
		const CostFunctionGradient: number =
			(2 / networkInputsLength) *
			networkOutputs.reduce(
				(prev, curr, i) => prev + (curr - targetOutputs[i])
			) *
			sigmoid(weightedSum) *
			(1 - sigmoid(weightedSum)) *
			networkInput;
		console.log(
			`GRADIENTS_ Cost function with respect to weights[i]: ${CostFunctionGradient}`
		);
		return CostFunctionGradient;
	}

	// PD of C / PD of b = 2/n * SUM(NO[i] - PO[i]) * S(z) * (1 - S(z))
	public Biases(
		networkOutputs: number[],
		weightedSum: number,
		targetOutputs: number[],
		networkInputsLength: number
	): number {
		const CostFunctionGradient: number =
			(2 / networkInputsLength) *
			networkOutputs.reduce(
				(prev, curr, i) => prev + (curr - targetOutputs[i])
			) *
			sigmoid(weightedSum) *
			(1 - sigmoid(weightedSum));
		console.log(
			`GRADIENTS_ Cost function with respect to bias: ${CostFunctionGradient}`
		);
		return CostFunctionGradient;
	}
}

export class GradientDescent {
	// w[i] = w[i] - (LR * GW)
	public UpdateWeights(
		neuron: Neuron,
		learnRate: number,
		networkOutputs: number[],
		weightedSum: number,
		networkInput: number,
		targetOutputs: number[],
		networkInputsLength: number
	): void {
		for (let i = 0; i < neuron.weights.length; i++) {
			const oldWeight: number = neuron.weights[i];
			neuron.weights[i] =
				neuron.weights[i] -
				learnRate *
					new Gradients().Weights(
						networkOutputs,
						weightedSum,
						networkInput,
						targetOutputs,
						networkInputsLength
					);
			console.log(
				`OPTIMIZATION_ weight changed (${i + 1}/${
					neuron.weights.length
				}): ${oldWeight} => ${neuron.weights[i]}`
			);
		}
	}

	// b = b - (LR * GB)
	public UpdateBias(
		neuron: Neuron,
		learnRate: number,
		networkOutputs: number[],
		weightedSum: number,
		networkInput: number,
		targetOutputs: number[],
		networkInputsLength: number
	): void {
		const oldBias: number = neuron.bias;
		neuron.bias =
			neuron.bias -
			learnRate *
				new Gradients().Biases(
					networkOutputs,
					weightedSum,
					targetOutputs,
					networkInputsLength
				);
		console.log(`OPTIMIZATION_ bias changed: ${oldBias} => ${neuron.bias}`);
	}
}
