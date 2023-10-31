import { data as trainingData } from "./Config/Data/dataset.json"; // XOR training dataset
import { learnCycles, learnRate } from "./Config/config.json"; // training config

import { NeuralNetwork } from "./src/Network/NeuralNetwork";

const NN = new NeuralNetwork(2, 1, 1);
