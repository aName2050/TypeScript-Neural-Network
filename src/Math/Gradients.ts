import { Activation } from '../Math/Activation';
const sigmoid = new Activation().Sigmoid;

export class GradientDescent {
    // Partial derivative of cost with respect to predicted output (PO)
    private partialCostPartialPO(
        networkOutput: number,
        targetOutput: number
    ): number {
        return 2 * (networkOutput - targetOutput);
    }

    // Partial derivative of predicted output (PO) with respect to weighted sum (z)
    private partialPOPartialZ(z: number): number {
        const sigmoidZ: number = sigmoid(z);
        return sigmoidZ * (1 - sigmoidZ);
    }

    // Partial derivative of weighted sum (z) with respect to weight (w[i])
    private partialZPartialWi(networkInput: number): number {
        return networkInput;
    }

    // Partial derivative of cost with respect to weight (w[i])
    public partialCostPartialWi(
        networkOutput: number,
        targetOutput: number,
        z: number,
        input: number
    ): number {
        const dC_dPO: number = this.partialCostPartialPO(
            networkOutput,
            targetOutput
        );
        const dPO_dZ: number = this.partialPOPartialZ(z);
        const dZ_dWi: number = this.partialZPartialWi(input);
        return dC_dPO * dPO_dZ * dZ_dWi;
    }

    // Partial derivative of cost with respect to bias (b)
    public partialCostPartialB(
        networkOutput: number,
        targetOutput: number,
        z: number
    ): number {
        const dC_dPO: number = this.partialCostPartialPO(
            networkOutput,
            targetOutput
        );
        const dPO_dZ: number = this.partialPOPartialZ(z);
        return dC_dPO * dPO_dZ;
    }

    // Update weight
    public UpdateWeight(
        weight: number,
        learnRate: number,
        gradient: number
    ): number {
        return weight - learnRate * gradient;
    }

    // Update bias
    public UpdateBias(
        bias: number,
        learnRate: number,
        gradient: number
    ): number {
        return bias - learnRate * gradient;
    }
}
