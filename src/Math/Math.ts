/**
 *
 * @param max The max number for random number generation
 * @param min The min number for random number generation
 * @returns The random number
 */
export function Random(max: number, min: number): number {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.random() * (max - min + 1) + min;
}
