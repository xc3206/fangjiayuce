// 交叉验证工具
export interface ValidationResult {
  rmse: number;
  mae: number;
  r2: number;
  predictions: number[];
  actual: number[];
}

export interface CrossValidationResult {
  foldResults: ValidationResult[];
  averageRMSE: number;
  averageMAE: number;
  averageR2: number;
  standardDeviationRMSE: number;
}

// 计算均方根误差
export function calculateRMSE(actual: number[], predicted: number[]): number {
  const mse = actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0) / actual.length;
  return Math.sqrt(mse);
}

// 计算平均绝对误差
export function calculateMAE(actual: number[], predicted: number[]): number {
  return actual.reduce((sum, val, i) => sum + Math.abs(val - predicted[i]), 0) / actual.length;
}

// 计算R²分数
export function calculateR2(actual: number[], predicted: number[]): number {
  const meanActual = actual.reduce((sum, val) => sum + val, 0) / actual.length;
  const totalSumSquares = actual.reduce((sum, val) => sum + Math.pow(val - meanActual, 2), 0);
  const residualSumSquares = actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0);
  
  if (totalSumSquares === 0) return 0;
  return 1 - (residualSumSquares / totalSumSquares);
}

// 五折交叉验证
export function kFoldCrossValidation(
  X: number[][],
  y: number[],
  modelTrainer: (trainX: number[][], trainY: number[]) => any,
  k: number = 5
): CrossValidationResult {
  const n = X.length;
  const foldSize = Math.floor(n / k);
  const indices = Array.from({ length: n }, (_, i) => i);
  
  // 打乱索引
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  const foldResults: ValidationResult[] = [];
  
  for (let fold = 0; fold < k; fold++) {
    const testStart = fold * foldSize;
    const testEnd = fold === k - 1 ? n : testStart + foldSize;
    
    // 分割训练集和测试集
    const testIndices = indices.slice(testStart, testEnd);
    const trainIndices = [...indices.slice(0, testStart), ...indices.slice(testEnd)];
    
    const trainX = trainIndices.map(i => X[i]);
    const trainY = trainIndices.map(i => y[i]);
    const testX = testIndices.map(i => X[i]);
    const testY = testIndices.map(i => y[i]);
    
    // 训练模型
    const model = modelTrainer(trainX, trainY);
    
    // 预测
    const predictions = model.predictBatch(testX);
    
    // 计算指标
    const rmse = calculateRMSE(testY, predictions);
    const mae = calculateMAE(testY, predictions);
    const r2 = calculateR2(testY, predictions);
    
    foldResults.push({
      rmse,
      mae,
      r2,
      predictions,
      actual: testY
    });
  }
  
  // 计算平均指标
  const averageRMSE = foldResults.reduce((sum, result) => sum + result.rmse, 0) / k;
  const averageMAE = foldResults.reduce((sum, result) => sum + result.mae, 0) / k;
  const averageR2 = foldResults.reduce((sum, result) => sum + result.r2, 0) / k;
  
  // 计算标准差
  const rmseVariance = foldResults.reduce((sum, result) => sum + Math.pow(result.rmse - averageRMSE, 2), 0) / k;
  const standardDeviationRMSE = Math.sqrt(rmseVariance);
  
  return {
    foldResults,
    averageRMSE,
    averageMAE,
    averageR2,
    standardDeviationRMSE
  };
}