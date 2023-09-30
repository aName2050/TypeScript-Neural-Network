import { data as Dataset } from '../Config/Data/dataset.json';
import { SummationOfMSE } from '../Utilities/Summation';

function Train(actualValue: number) {
    // LEARNING ALGORITHM
    // Cost Function: calculate with MSE with every value in the dataset
    // lower is better
    var Cost = (1 / Dataset.length) * SummationOfMSE(Dataset, [actualValue]);
    console.log('cost: ' + Cost);
    // gradients (rate of change)
}
