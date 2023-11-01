export class Calculus {
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
        });
        return result;
    }
}
