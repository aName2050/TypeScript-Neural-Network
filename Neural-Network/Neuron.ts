class Neuron {
    public weights: number[];
    public bias: number;

    constructor(numInputs: number) {
        this.weights = [];
        for (let i = 0; i < numInputs; i++) {
            this.weights.push(Math.random());
        }
        this.bias = Math.random();
    }
}
