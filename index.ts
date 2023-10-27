import { data as trainingData } from "./Config/Data/dataset.json"; // XOR training dataset
import { learnCycles, learnRate } from "./Config/config.json"; // training config
/*
Layer size needs to be fewer than the previous layer size, excluding input layer
[inputLayerSize, ...hiddenLayerSizes < prevLayerSize, outputLayer < prevHiddenLayerSize]
*/
import { NeuralNetwork } from "./Neural-Network/NeuralNetwork";
import { Neuron } from "./Neural-Network/Neuron";
console.log(`TRAINING-DATA:`, trainingData);

const neuralNetwork: NeuralNetwork = new NeuralNetwork([2, 1, 1]);
const neuron: Neuron = new Neuron(2, false);
// Training
console.log(`TRAINING_ training begun; ${learnCycles} training cycles`);
for (let i = 0; i < learnCycles; i++) {
	for (let j = 0; j < trainingData.length; j++) {
		const data = trainingData[j];

		console.log(
			`TRAINING_ cycle: ${i + 1}/${learnCycles} training-data: ${j + 1}/${
				trainingData.length
			}`
		);
		console.log(
			`TRAINING_ EXPECTED... input: ${data.input} output: ${data.output}`
		);

		neuralNetwork.train(data.input, data.output, learnRate);
	}
}
// Testing
const testInput: number[] = [0, 0];
console.log(`TESTING_ input:`, testInput);

const output: number[] = neuralNetwork.forwardPropagation(testInput);
console.log("Neural Network Output:", output);
