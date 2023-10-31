export class Calculus {
	/**
	 *
	 * @param inputs The inputs to the summation function
	 * @returns The result of the summation function
	 */
	public Summation(inputs: number[]): number {
		let result: number = 0;
		inputs.forEach((input) => {
			result += input;
		});
		return result;
	}
}
