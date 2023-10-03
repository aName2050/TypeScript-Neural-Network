interface IActivation {
    Activate(weightedInputs: number[], index: number): number;
    Derivative(weightedInputs: number[], index: number): number;
}

class Layer {
    public readonly numNodesIn: number;
    public readonly numNodesOut: number;
    public readonly weights: number[];
    public readonly biases: number[];
    public readonly costGradientW: number[];
    public readonly costGradientB: number[];
    public readonly weightVelocities: number[];
    public readonly biasVelocities: number[];
    public activation: IActivation;

    constructor(numNodesIn: number, numNodesOut: number, rng: any) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        this.activation = new Sigmoid();

        this.weights = new Array(numNodesIn * numNodesOut);
        this.costGradientW = new Array(this.weights.length).fill(0);
        this.biases = new Array(numNodesOut);
        this.costGradientB = new Array(this.biases.length).fill(0);

        this.weightVelocities = new Array(this.weights.length).fill(0);
        this.biasVelocities = new Array(this.biases.length).fill(0);

        this.initializeRandomWeights(rng);
    }

    public calculateOutputs(inputs: number[]): number[] {
        const weightedInputs: number[] = new Array(this.numNodesOut).fill(0);

        for (let nodeOut = 0; nodeOut < this.numNodesOut; nodeOut++) {
            weightedInputs[nodeOut] = this.biases[nodeOut];
            for (let nodeIn = 0; nodeIn < this.numNodesIn; nodeIn++) {
                weightedInputs[nodeOut] +=
                    inputs[nodeIn] * this.getWeight(nodeIn, nodeOut);
            }
        }

        const activations: number[] = new Array(this.numNodesOut);
        for (let outputNode = 0; outputNode < this.numNodesOut; outputNode++) {
            activations[outputNode] = this.activation.Activate(
                weightedInputs,
                outputNode
            );
        }

        return activations;
    }

    public applyGradients(
        learnRate: number,
        regularization: number,
        momentum: number
    ): void {
        const weightDecay = 1 - regularization * learnRate;

        for (let i = 0; i < this.weights.length; i++) {
            const weight = this.weights[i];
            const velocity =
                this.weightVelocities[i] * momentum -
                this.costGradientW[i] * learnRate;
            this.weightVelocities[i] = velocity;
            this.weights[i] = weight * weightDecay + velocity;
            this.costGradientW[i] = 0;
        }

        for (let i = 0; i < this.biases.length; i++) {
            const velocity =
                this.biasVelocities[i] * momentum -
                this.costGradientB[i] * learnRate;
            this.biasVelocities[i] = velocity;
            this.biases[i] += velocity;
            this.costGradientB[i] = 0;
        }
    }

    private getWeight(nodeIn: number, nodeOut: number): number {
        const flatIndex = nodeOut * this.numNodesIn + nodeIn;
        return this.weights[flatIndex];
    }

    private initializeRandomWeights(rng: any): void {
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] =
                this.randomInNormalDistribution(rng, 0, 1) /
                Math.sqrt(this.numNodesIn);
        }
    }

    private randomInNormalDistribution(
        rng: any,
        mean: number,
        standardDeviation: number
    ): number {
        const x1 = 1 - rng.next();
        const x2 = 1 - rng.next();

        const y1 =
            Math.sqrt(-2.0 * Math.log(x1)) * Math.cos(2.0 * Math.PI * x2);
        return y1 * standardDeviation + mean;
    }
}
