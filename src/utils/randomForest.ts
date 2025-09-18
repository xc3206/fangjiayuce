// 简化的随机森林实现
export class DecisionTree {
  private feature: number = -1;
  private threshold: number = 0;
  private value: number = 0;
  private left: DecisionTree | null = null;
  private right: DecisionTree | null = null;
  private isLeaf: boolean = true;

  train(X: number[][], y: number[], maxDepth: number = 10, minSamples: number = 5) {
    if (X.length <= minSamples || maxDepth <= 0) {
      this.value = y.reduce((sum, val) => sum + val, 0) / y.length;
      this.isLeaf = true;
      return;
    }

    const bestSplit = this.findBestSplit(X, y);
    
    if (bestSplit.gain <= 0) {
      this.value = y.reduce((sum, val) => sum + val, 0) / y.length;
      this.isLeaf = true;
      return;
    }

    this.feature = bestSplit.feature;
    this.threshold = bestSplit.threshold;
    this.isLeaf = false;

    const { leftX, leftY, rightX, rightY } = this.splitData(X, y, bestSplit.feature, bestSplit.threshold);

    this.left = new DecisionTree();
    this.right = new DecisionTree();
    
    this.left.train(leftX, leftY, maxDepth - 1, minSamples);
    this.right.train(rightX, rightY, maxDepth - 1, minSamples);
  }

  private findBestSplit(X: number[][], y: number[]) {
    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;

    for (let feature = 0; feature < X[0].length; feature++) {
      const values = X.map(row => row[feature]);
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        const gain = this.calculateGain(X, y, feature, threshold);

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feature;
          bestThreshold = threshold;
        }
      }
    }

    return { gain: bestGain, feature: bestFeature, threshold: bestThreshold };
  }

  private calculateGain(X: number[][], y: number[], feature: number, threshold: number) {
    const { leftY, rightY } = this.splitData(X, y, feature, threshold);
    
    if (leftY.length === 0 || rightY.length === 0) return 0;

    const parentVar = this.variance(y);
    const leftVar = this.variance(leftY);
    const rightVar = this.variance(rightY);
    
    const n = y.length;
    const nLeft = leftY.length;
    const nRight = rightY.length;

    const weightedVar = (nLeft / n) * leftVar + (nRight / n) * rightVar;
    
    return parentVar - weightedVar;
  }

  private splitData(X: number[][], y: number[], feature: number, threshold: number) {
    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return { leftX, leftY, rightX, rightY };
  }

  private variance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return variance;
  }

  predict(x: number[]): number {
    if (this.isLeaf) {
      return this.value;
    }

    if (x[this.feature] <= this.threshold) {
      return this.left!.predict(x);
    } else {
      return this.right!.predict(x);
    }
  }
}

export class RandomForest {
  private trees: DecisionTree[] = [];
  private nTrees: number;

  constructor(nTrees: number = 10) {
    this.nTrees = nTrees;
  }

  train(X: number[][], y: number[]) {
    this.trees = [];
    
    for (let i = 0; i < this.nTrees; i++) {
      // Bootstrap sampling
      const { sampledX, sampledY } = this.bootstrap(X, y);
      
      const tree = new DecisionTree();
      tree.train(sampledX, sampledY, 10, 5);
      this.trees.push(tree);
    }
  }

  private bootstrap(X: number[][], y: number[]) {
    const n = X.length;
    const sampledX: number[][] = [];
    const sampledY: number[] = [];

    for (let i = 0; i < n; i++) {
      const index = Math.floor(Math.random() * n);
      sampledX.push([...X[index]]);
      sampledY.push(y[index]);
    }

    return { sampledX, sampledY };
  }

  predict(x: number[]): number {
    const predictions = this.trees.map(tree => tree.predict(x));
    return predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
  }

  predictBatch(X: number[][]): number[] {
    return X.map(x => this.predict(x));
  }
}