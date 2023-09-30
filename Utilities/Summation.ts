export function SummationOfValueAndWeight(array: number[][]): number {
    // input (x by 1 matrix): [[x1, w1], ..., [xN, wN]]
    var result: number = 0;
    for (var i = 0; i < array.length; i++) {
        result = result + array[i][0] * array[i][1];
    }
    return result;
}
export function SummationOfMSE(
    datasetArray: number[],
    networkArray: number[]
): number {
    // mean square error
    var MSE = function (yI: number, yi: number): number {
        return Math.pow(yI - yi, 2);
    };
    var result = 0;
    for (var i = 0; i < datasetArray.length; i++) {
        result = result + MSE(datasetArray[i], networkArray[i]);
    }
    return result;
}
