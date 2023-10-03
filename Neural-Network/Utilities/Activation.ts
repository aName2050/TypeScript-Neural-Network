class Activation {
    static getActivationFromType(type: ActivationType): IActivation {
        switch (type) {
            case ActivationType.Sigmoid:
                return new Sigmoid();
            case ActivationType.TanH:
                return new TanH();
            case ActivationType.ReLU:
                return new ReLU();
            case ActivationType.SiLU:
                return new SiLU();
            case ActivationType.Softmax:
                return new Softmax();
            default:
                console.error('Unhandled activation type');
                return new Sigmoid();
        }
    }
}

enum ActivationType {
    Sigmoid,
    TanH,
    ReLU,
    SiLU,
    Softmax,
}

interface IActivation {
    Activate(inputs: number[], index: number): number;
    Derivative(inputs: number[], index: number): number;
    GetActivationType(): ActivationType;
}

class Sigmoid implements IActivation {
    Activate(inputs: number[], index: number): number {
        return 1.0 / (1 + Math.exp(-inputs[index]));
    }

    Derivative(inputs: number[], index: number): number {
        let a = this.Activate(inputs, index);
        return a * (1 - a);
    }

    GetActivationType(): ActivationType {
        return ActivationType.Sigmoid;
    }
}

class TanH implements IActivation {
    Activate(inputs: number[], index: number): number {
        let e2 = Math.exp(2 * inputs[index]);
        return (e2 - 1) / (e2 + 1);
    }

    Derivative(inputs: number[], index: number): number {
        let e2 = Math.exp(2 * inputs[index]);
        let t = (e2 - 1) / (e2 + 1);
        return 1 - t * t;
    }

    GetActivationType(): ActivationType {
        return ActivationType.TanH;
    }
}

class ReLU implements IActivation {
    Activate(inputs: number[], index: number): number {
        return Math.max(0, inputs[index]);
    }

    Derivative(inputs: number[], index: number): number {
        return inputs[index] > 0 ? 1 : 0;
    }

    GetActivationType(): ActivationType {
        return ActivationType.ReLU;
    }
}

class SiLU implements IActivation {
    Activate(inputs: number[], index: number): number {
        return inputs[index] / (1 + Math.exp(-inputs[index]));
    }

    Derivative(inputs: number[], index: number): number {
        let sig = 1 / (1 + Math.exp(-inputs[index]));
        return inputs[index] * sig * (1 - sig) + sig;
    }

    GetActivationType(): ActivationType {
        return ActivationType.SiLU;
    }
}

class Softmax implements IActivation {
    Activate(inputs: number[], index: number): number {
        let expSum = 0;
        for (let i = 0; i < inputs.length; i++) {
            expSum += Math.exp(inputs[i]);
        }
        let res = Math.exp(inputs[index]) / expSum;
        return res;
    }

    Derivative(inputs: number[], index: number): number {
        let expSum = 0;
        for (let i = 0; i < inputs.length; i++) {
            expSum += Math.exp(inputs[i]);
        }
        let ex = Math.exp(inputs[index]);
        return (ex * expSum - ex * ex) / (expSum * expSum);
    }

    GetActivationType(): ActivationType {
        return ActivationType.Softmax;
    }
}
