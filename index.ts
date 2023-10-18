import { data as trainingData } from "./Config/Data/dataset.json";
/*
Layer size needs to be fewer than the previous layer size, excluding input layer
[inputLayerSize, ...hiddenLayerSizes, outputLayer]
*/
import { NeuralNetwork } from "./Neural-Network/NeuralNetwork";
console.log(`training data:`, trainingData);

const neuralNetwork = new NeuralNetwork([2, 1, 1]);
for (let i = 0; i < 2; i++) {
  for (const data of trainingData) {
    neuralNetwork.train(data.input, data.output);

    console.log(`TRAINING... #${i} => IN: ${data.input} OUT: ${data.output}`);
  }
}
const testInput = [1, 0];
const output = neuralNetwork.forwardPropagation(testInput);
console.log("Neural Network Output:", output); // Output: [1]
