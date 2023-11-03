import { data as trainingData } from "./Config/Data/dataset.json"; // XOR training dataset
import { learnCycles, learnRate } from "./Config/config.json"; // training config

import { NeuralNetwork } from "./src/Network/NeuralNetwork";

const NN: NeuralNetwork = new NeuralNetwork(2, 1, 1);
// NN.forwardPropagation([0, 0]);

for(let i=0;i<learnCycles;i++){
    for(let j=0;j<trainingData.length;j++){
        NN.train(trainingData[0].input, trainingData[0].output, learnRate);
    }
}

// NN.forwardPropagation([0, 0]);
