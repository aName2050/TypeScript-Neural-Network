export class Activation {
    /**
     *
     * @param x The input of the function
     * @returns The output of the function
     */
    public Sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }
}
