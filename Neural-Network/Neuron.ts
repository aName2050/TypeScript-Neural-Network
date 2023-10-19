export class Neuron {
  public weights: number[];
  public bias: number;

  constructor(numInputs: number) {
    console.log(`INIT-NEURON_ inputs: ${numInputs}`);
    this.weights = Array.from(
      { length: numInputs },
      () => Math.random() * 2 - 1
    );

    this.bias = Math.random() * 2 - 1;
  }
}
