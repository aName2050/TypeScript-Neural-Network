import { Neuron } from "./Neuron";

export class NeuralNetwork {
  public layers: Neuron[][];

  constructor(layerSizes: number[]) {
    this.layers = [];

    console.log(`init-layerSizes: ${layerSizes}`);

    // Input layer
    this.layers.push(
      Array.from({ length: layerSizes[0] }, () => new Neuron(1))
    );

    // Hidden layers
    for (let i = 1; i < layerSizes.length; i++) {
      console.log(`hidden-layer-init_sizes: ${layerSizes[i]}`);
      this.layers.push(
        Array.from(
          { length: layerSizes[i] },
          () => new Neuron(layerSizes[i - 1])
        )
      );
    }

    // Output Layer
    this.layers.push(
      Array.from(
        { length: layerSizes[layerSizes.length - 1] },
        () => new Neuron(layerSizes[layerSizes.length - 2])
      )
    );
  }

  public forwardPropagation(inputs: number[]): number[] {
    let outputs = inputs;

    for (const layer of this.layers) {
      const newOutputs = [];
      for (const neuron of layer) {
        console.log(
          `fwdProp_ weights: ${neuron.weights}; bias: ${neuron.bias}`
        );
        let weightedSum = 0;
        for (let i = 0; i < neuron.weights.length; i++) {
          console.log(
            `fwdProp_ weightedSum_eqauation: (${i}) ${weightedSum} + ${neuron.weights[i]} * ${outputs[i]}`
          );

          weightedSum += neuron.weights[i] * outputs[i];
        }
        console.log(`fwdProp_ weightedSum: ${weightedSum}`);

        const activation = this.sigmoid(weightedSum + neuron.bias);
        newOutputs.push(activation);
      }
      outputs = newOutputs;
    }
    console.log(`fwdProp_ output:`, outputs);

    return outputs;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  public train(
    inputs: number[],
    targets: number[],
    learningRate: number = 0.1
  ): void {
    // Forward propagation
    const outputs = this.forwardPropagation(inputs);

    // Error calculation per neuron
    console.log(`TRAINING_ outputs:`, outputs);
    const outputErrors = [];
    for (let i = 0; i < outputs.length; i++) {
      console.log(
        `TRAINING_ERR_CALC_ target: i:(${i}) ${targets[i]} output: ${
          outputs[i]
        } error: ${targets[i] - outputs[i]}`
      );
      outputErrors.push(targets[i] - outputs[i]);
    }

    // Backpropagation learning algorithm
    // update biases and weights for output layer
    for (let i = 0; i < this.layers[this.layers.length - 1].length; i++) {
      const neuron = this.layers[this.layers.length - 1][i];
      if (outputErrors[i] == undefined)
        console.log(`TRAINING_ missing outputErrors[i:${i}]`);
      console.log(
        `TRAINING_ bias: ${neuron.bias} outputErrors: i:${outputErrors[i]} LR: ${learningRate}`
      );
      neuron.bias +=
        outputErrors[i] * this.sigmoidDerivative(outputs[i]) * learningRate;
      for (let j = 0; j < neuron.weights.length; j++) {
        if (outputErrors[i] == undefined)
          console.log(`TRAINING_ missing outputErrors[j:${j}]`);
        console.log(
          `TRAINING_ weight: j:${neuron.weights[j]} outputErrors: j:${outputErrors[j]} inputs: i:${inputs[j]} LR: ${learningRate}`
        );
        neuron.weights[j] +=
          outputErrors[i] *
          this.sigmoidDerivative(outputs[i]) *
          inputs[j] *
          learningRate;
      }
    }
    // update biases and weights for hidden layers
    for (let i = this.layers.length - 2; i > 0; i--) {
      for (const neuron of this.layers[i]) {
        let error = 0;
        for (let j = 0; j < this.layers[i + 1].length; j++) {
          console.log(
            `TRAINING_ bias: ${neuron.bias} outputErrors: i:${outputErrors[i]} LR: ${learningRate}`
          );
          console.log(
            `TRAINING_ weight: j:${neuron.weights[j]} outputErrors: i:${outputErrors[i]} inputs: j:${inputs[j]} LR: ${learningRate}`
          );
          error +=
            this.layers[i + 1][j].weights[this.layers[i].indexOf(neuron)] *
            outputErrors[j];
        }
        let weightedSum = 0;
        for (var k = 0; k < neuron.weights.length; k++) {
          weightedSum += neuron.weights[k] * inputs[k];
        }

        error *= this.sigmoidDerivative(weightedSum + neuron.bias);

        neuron.bias += error * learningRate;
        for (let j = 0; j < neuron.weights.length; j++) {
          neuron.weights[j] += error * inputs[j] * learningRate;
        }
      }
    }
  }

  private sigmoidDerivative(x: number): number {
    return this.sigmoid(x) * (1 - this.sigmoid(x));
  }
}
