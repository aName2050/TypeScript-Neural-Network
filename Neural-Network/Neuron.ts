import { SummationOfValueAndWeight } from '../Utilities/Summation';
import { Sigmoid } from '../Utilities/Sigmoid';
import CUID from 'cuid';

export class Neuron {
    static id: string;
    constructor(layer: string, inputs: number[][]) {
        Neuron.id = CUID(); // generate a random ID for this neuron

        // Weights and biases
        const weights: number[] = [1, 3, 5, 7];
        const bias: number = Math.floor(Math.random() * (5 - -5) + -5);

        // Matrix
        let Matrix: number[][] = [];
        if (inputs.length != weights.length)
            throw 'There must be the same amount of weights as inputs';
        // FIXME:
        // TODO:
        // inputs.forEach(input => {
        //     Matrix.push([inputs[i], weights[i]]);
        // });
    }
}

/** REFERENCE */

// var weights = [1, 3, 5, 7];
//   var bias = 0;
//   // generate matrix (1 row by (inputs/weights amount) columns)
//   var Matrix = [];
//   if(inputs.length != weights.length) throw 'There must be the same amount of weights as inputs';
//   for(var i=0;i<inputs.length;i++){
//     Matrix.push([inputs[i], weights[i]]);
//   }
//   // calculate dot product of inputs and weights
//   var dotProduct = SummationOfValueAndWeight(Matrix);
//   // add the bias to the dot product
//   var z = dotProduct + bias;
//   // activation function
//   var output = Sigmoid(z);
//   return output;
