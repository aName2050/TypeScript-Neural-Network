import { data as trainingData } from "./Config/Data/dataset.json"; // OR training dataset
import { learnCycles, learnRate } from "./Config/config.json"; // training config

import { NeuralNetwork } from "./src/Network/NeuralNetwork";

import chalk from "chalk";

const NN: NeuralNetwork = new NeuralNetwork(4, 3, 3, 2);
// Pre-training test
let preTrainTestResults: number[][] = [];
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[0].input} Expect: ~ ${trainingData[0].output} `
		)
);
preTrainTestResults.push(NN.forwardPropagation(trainingData[0].input)); // Expect horribly wrong (or right) answer
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[1].input} Expect: ~ ${trainingData[1].output} `
		)
);
preTrainTestResults.push(NN.forwardPropagation(trainingData[1].input)); // Expect horribly wrong (or right) answer
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[2].input} Expect: ~ ${trainingData[2].output} `
		)
);
preTrainTestResults.push(NN.forwardPropagation(trainingData[2].input)); // Expect horribly wrong (or right) answer
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[3].input} Expect: ~ ${trainingData[3].output} `
		)
);
preTrainTestResults.push(NN.forwardPropagation(trainingData[3].input)); // Expect horribly wrong (or right) answer

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
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[0].input} Expect: ~ ${trainingData[0].output} `
		)
);
NN.forwardPropagation(trainingData[0].input); // Expect a somewhat right answer (target 0)
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[1].input} Expect: ~ ${trainingData[1].output} `
		)
);
NN.forwardPropagation(trainingData[1].input); // Expect a somewhat right answer
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[2].input} Expect: ~ ${trainingData[2].output} `
		)
);
NN.forwardPropagation(trainingData[2].input); // Expect a somewhat right answer
console.log(
	chalk
		.bgKeyword("orange")
		.bold(
			`Input: ${trainingData[3].input} Expect: ~ ${trainingData[3].output} `
		)
);
NN.forwardPropagation(trainingData[3].input); // Expect a somewhat right answer

// Pre-training test
console.log(
	`${chalk.bold.bgCyan("PRE-TRAINING TEST...............................")}`
);
const displayTexts: string[] = [
	`Input: ${trainingData[0].input} Expect: ~ ${trainingData[0].output}`,
	`Input: ${trainingData[1].input} Expect: ~ ${trainingData[1].output}`,
	`Input: ${trainingData[2].input} Expect: ~ ${trainingData[2].output}`,
	`Input: ${trainingData[3].input} Expect: ~ ${trainingData[3].output}`,
];
preTrainTestResults.forEach((result, i) => {
	console.log(
		chalk.bgKeyword("pink").bold(`${displayTexts[i]} Result: ${result} `)
	);
});
