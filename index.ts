import { data as trainingData } from './Config/Data/dataset.json';
import { NeuralNetwork } from './Neural-Network/NeuralNetwork';
console.log(`training data:`, trainingData);

const neuralNetwork = new NeuralNetwork([2, 3, 1]);
for (let i = 0; i < 10000; i++) {
    for (const data of trainingData) {
        neuralNetwork.train(data.input, data.output);
        console.log(
            `TRAINING... #${i} => IN: ${data.input} OUT: ${data.output}`
        );
    }
}
const testInput = [1, 0];
const output = neuralNetwork.forwardPropagation(testInput);
console.log('Neural Network Output:', output); // Output: [1]
