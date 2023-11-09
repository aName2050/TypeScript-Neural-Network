import { data as trainingData } from "./Config/Data/dataset.json"; // OR training dataset
import { learnCycles, learnRate } from "./Config/config.json"; // training config

import { NeuralNetwork } from "./src/Network/NeuralNetwork";

import chalk from "chalk";

const NN: NeuralNetwork = new NeuralNetwork(4, 1, 1, 2);
// Pre-training test
let preTrainTestResults: number[][] = [];
for (let i = 0; i < trainingData.length; i++) {
	console.log(
		chalk
			.bgKeyword("orange")
			.bold(
				`Input: ${trainingData[i].input} Expect: ~ ${trainingData[i].output} `
			)
	);
	preTrainTestResults.push(NN.forwardPropagation(trainingData[i].input)); // Expect horribly wrong (or right) answer
}

// Train network
console.log(
	chalk.bold.bgMagentaBright(
		"TRAINING NETWORK..............................."
	)
);
const trainStart: number = Date.now();
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
console.log(
	chalk.bold.bgCyan("POST-TRAINING INFO...............................")
);
console.log(`Training took ${Date.now() - trainStart}ms`);
console.log(`LearnCycles: ${learnCycles}`);
console.log(`LearnRate: ${learnRate}`);
console.log(`TrainingData.length: ${trainingData.length}`);

// Post-training test
console.log(
	`${chalk.bold.bgCyan("POST-TRAINING TEST...............................")}`
);
for (let i = 0; i < trainingData.length; i++) {
	console.log(
		chalk
			.bgKeyword("orange")
			.bold(
				`Input: ${trainingData[i].input} Expect: ~ ${trainingData[i].output} `
			)
	);
	NN.forwardPropagation(trainingData[i].input); // Expect a somewhat right answer
}

// Pre-training test
console.log(
	`${chalk.bold.bgCyan("PRE-TRAINING TEST...............................")}`
);
const displayTexts: string[] = Array.from(
	{ length: trainingData.length },
	(v, k) =>
		`Input: ${trainingData[k].input} Expect ~ ${trainingData[k].output}`
);
preTrainTestResults.forEach((result, i) => {
	console.log(
		chalk.bgKeyword("pink").bold(`${displayTexts[i]} Result: ${result} `)
	);
});
