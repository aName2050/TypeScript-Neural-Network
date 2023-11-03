import { data as trainingData } from "./Config/Data/dataset.json"; // XOR training dataset
import { learnCycles, learnRate } from "./Config/config.json"; // training config

import { NeuralNetwork } from "./src/Network/NeuralNetwork";

import chalk from "chalk";

const NN: NeuralNetwork = new NeuralNetwork(2, 1, 1);
// Pre-training test
console.log(
	`${chalk.bold.bgCyanBright(
		"PRE-TRAINING TEST..............................."
	)}`
);
NN.forwardPropagation([0, 0]); // Expect horribly wrong (or right) answer (target 0)

// Train network
console.log(
	chalk.bold.bgMagentaBright(
		"TRAINING NETWORK..............................."
	)
);
for (let i = 0; i < learnCycles; i++) {
	for (let j = 0; j < trainingData.length; j++) {
		console.log(
			`${chalk.bold.bgRedBright("TRAINING_")} cycles (${
				i + 1
			}/${learnCycles}) trainingData (${j + 1}/${
				trainingData.length
			}) learnRate: ${learnRate} in: ${
				trainingData[j].input
			} expectOut: ${trainingData[j].output}`
		);
		NN.train(trainingData[j].input, trainingData[j].output, learnRate);
	}
}

// Post-training test
console.log(
	`${chalk.bold.bgCyanBright(
		"POST-TRAINING TEST..............................."
	)}`
);
NN.forwardPropagation([0, 0]); // Expect a somewhat right answer (target 0)
