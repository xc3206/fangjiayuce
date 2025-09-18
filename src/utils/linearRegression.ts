// 线性回归模型实现
export class LinearRegression {
  private weights: number[] = [];
  private bias: number = 0;
  private featureCount: number = 0;

  // 训练模型
  train(X: number[][], y: number[], learningRate: number = 0.0001, epochs: number = 1000) {
    const m = X.length;
    this.featureCount = X[0].length;
    
    // 初始化权重和偏置
    this.weights = new Array(this.featureCount).fill(0);
    this.bias = 0;
    
    // 梯度下降
    for (let epoch = 0; epoch < epochs; epoch++) {
      // 前向传播
      const predictions = X.map(sample => this.predict(sample));
      
      // 计算梯度
      const weightGradients = new Array(this.featureCount).fill(0);
      let biasGradient = 0;
      
      for (let i = 0; i < m; i++) {
        const error = predictions[i] - y[i];
        biasGradient += error;
        
        for (let j = 0; j < this.featureCount; j++) {
          weightGradients[j] += error * X[i][j];
        }
      }
      
      // 更新参数
      for (let j = 0; j < this.featureCount; j++) {
        this.weights[j] -= learningRate * weightGradients[j] / m;
      }
      this.bias -= learningRate * biasGradient / m;
    }
  }
  
  // 预测
  predict(x: number[]): number {
    let result = this.bias;
    for (let i = 0; i < this.featureCount; i++) {
      result += this.weights[i] * x[i];
    }
    return result;
  }
  
  // 批量预测
  predictBatch(X: number[][]): number[] {
    return X.map(x => this.predict(x));
  }
  
  getWeights(): number[] {
    return [...this.weights];
  }
  
  getBias(): number {
    return this.bias;
  }
}