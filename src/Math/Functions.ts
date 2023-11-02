export class Functions {
	/**
	 * This is also considered the Dot Product of the neuron
	 *
	 * @param inputs The inputs to the summation function
	 * @returns The result of the summation function
	 */
	public WeightedSumCalculation(inputs: number[], weights: number[]): number {
		let result: number = 0;
		inputs.forEach((input, i) => {
			result += input * weights[i];
			console.log(`z = ${result} + (${input} * ${weights[i]})`);
		});
		return result;
	}

	/**
	 * Loss function
	 * @param network The network's output
	 * @param dataset The expected output
	 * @returns The loss of the network
	 */
	private MSE(network: number, dataset: number): number {
		return Math.pow(network - dataset, 2);
	}

	/**
	 *
	 * @param networkOutputs The network's outputs
	 * @param expectedOutputs The expected outputs
	 * @returns The average cost (loss) of the network
	 */
	public Cost(networkOutputs: number[], expectedOutputs: number[]): number {
		let MSEOutputs: number[] = [];
		for (let i = 0; i < networkOutputs.length; i++) {
			const MSEOutput: number = this.MSE(
				networkOutputs[i],
				expectedOutputs[i]
			);
			MSEOutputs.push(MSEOutput);
		}
		const loss: number = MSEOutputs.reduce((prev, curr) => prev + curr);
		const cost: number = (1 / expectedOutputs.length) * loss;

		console.log(`TRAINING_ network cost: ${cost} loss: ${loss}`);
		return cost;
	}
}
