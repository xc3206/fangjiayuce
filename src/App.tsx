import React, { useState, useEffect } from 'react';
import { TrendingUp, Database, BarChart3, Brain } from 'lucide-react';
import { generateHouseData, HouseData } from './utils/dataGenerator';
import { LinearRegression } from './utils/linearRegression';
import { RandomForest } from './utils/randomForest';
import { kFoldCrossValidation, CrossValidationResult } from './utils/crossValidation';
import { ModelPerformanceChart } from './components/ModelPerformanceChart';
import { PredictionForm, PredictionInput } from './components/PredictionForm';

interface ModelResults {
  [key: string]: {
    model: any;
    cvResults: CrossValidationResult;
  };
}

function App() {
  const [houseData, setHouseData] = useState<HouseData[]>([]);
  const [modelResults, setModelResults] = useState<ModelResults>({});
  const [predictions, setPredictions] = useState<{ [key: string]: number }>({});
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('线性回归');

  // 特征提取函数
  const extractFeatures = (data: HouseData[]): number[][] => {
    return data.map(house => [
      house.area,
      house.bedrooms,
      house.bathrooms,
      house.age,
      house.floor,
      house.hasParking,
      house.hasGarden,
      house.distanceToCenter
    ]);
  };

  const extractTargets = (data: HouseData[]): number[] => {
    return data.map(house => house.price);
  };

  // 生成数据并训练模型
  const generateAndTrainModels = async () => {
    setIsTraining(true);
    
    // 生成房屋数据
    const data = generateHouseData(1000);
    setHouseData(data);
    
    const X = extractFeatures(data);
    const y = extractTargets(data);
    
    const results: ModelResults = {};
    
    // 训练线性回归模型
    const lrCvResults = kFoldCrossValidation(X, y, (trainX, trainY) => {
      const model = new LinearRegression();
      model.train(trainX, trainY, 0.00001, 2000);
      return model;
    });
    
    // 训练最终的线性回归模型（用于预测）
    const finalLrModel = new LinearRegression();
    finalLrModel.train(X, y, 0.00001, 2000);
    
    results['线性回归'] = {
      model: finalLrModel,
      cvResults: lrCvResults
    };
    
    // 训练随机森林模型
    const rfCvResults = kFoldCrossValidation(X, y, (trainX, trainY) => {
      const model = new RandomForest(10);
      model.train(trainX, trainY);
      return model;
    });
    
    // 训练最终的随机森林模型（用于预测）
    const finalRfModel = new RandomForest(10);
    finalRfModel.train(X, y);
    
    results['随机森林'] = {
      model: finalRfModel,
      cvResults: rfCvResults
    };
    
    setModelResults(results);
    setIsTraining(false);
  };

  // 预测房价
  const handlePredict = (input: PredictionInput) => {
    const features = [
      input.area,
      input.bedrooms,
      input.bathrooms,
      input.age,
      input.floor,
      input.hasParking ? 1 : 0,
      input.hasGarden ? 1 : 0,
      input.distanceToCenter
    ];

    const newPredictions: { [key: string]: number } = {};
    
    Object.entries(modelResults).forEach(([modelName, { model }]) => {
      newPredictions[modelName] = model.predict(features);
    });
    
    setPredictions(newPredictions);
  };

  // 组件加载时自动生成数据和训练模型
  useEffect(() => {
    generateAndTrainModels();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-8 h-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-800">
                智能房价预测系统
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-sm text-gray-600">
                <Database className="w-4 h-4 mr-1" />
                数据量: {houseData.length}
              </div>
              <button
                onClick={generateAndTrainModels}
                disabled={isTraining}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {isTraining ? '训练中...' : '重新训练'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {isTraining ? (
          <div className="text-center py-20">
            <Brain className="w-16 h-16 text-blue-600 mx-auto mb-4 animate-pulse" />
            <h2 className="text-2xl font-bold text-gray-800 mb-2">正在训练模型...</h2>
            <p className="text-gray-600">正在生成数据并训练机器学习模型，请稍候...</p>
          </div>
        ) : (
          <div className="space-y-8">
            {/* 模型选择和预测表单 */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-1">
                <PredictionForm
                  onPredict={handlePredict}
                  predictions={predictions}
                />
              </div>
              
              {/* 数据统计 */}
              <div className="lg:col-span-2">
                <div className="bg-white p-6 rounded-lg shadow-md">
                  <div className="flex items-center mb-4">
                    <BarChart3 className="w-6 h-6 text-green-600 mr-2" />
                    <h2 className="text-xl font-bold text-gray-800">数据集统计</h2>
                  </div>
                  
                  {houseData.length > 0 && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {Math.round(houseData.reduce((sum, h) => sum + h.area, 0) / houseData.length)}
                        </div>
                        <div className="text-sm text-blue-500">平均面积 (m²)</div>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">
                          {Math.round(houseData.reduce((sum, h) => sum + h.price, 0) / houseData.length)}
                        </div>
                        <div className="text-sm text-green-500">平均价格 (万元)</div>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          {Math.round((houseData.filter(h => h.hasParking).length / houseData.length) * 100)}%
                        </div>
                        <div className="text-sm text-purple-500">配备停车位</div>
                      </div>
                      <div className="text-center p-4 bg-orange-50 rounded-lg">
                        <div className="text-2xl font-bold text-orange-600">
                          {Math.round(houseData.reduce((sum, h) => sum + h.age, 0) / houseData.length)}
                        </div>
                        <div className="text-sm text-orange-500">平均房龄 (年)</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* 模型性能对比 */}
            {Object.keys(modelResults).length > 0 && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-gray-800">模型性能对比</h2>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    {Object.keys(modelResults).map(modelName => (
                      <option key={modelName} value={modelName}>
                        {modelName}
                      </option>
                    ))}
                  </select>
                </div>

                {/* 模型性能表格 */}
                <div className="bg-white p-6 rounded-lg shadow-md overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-4 font-semibold text-gray-800">模型</th>
                        <th className="text-center py-3 px-4 font-semibold text-gray-800">平均RMSE</th>
                        <th className="text-center py-3 px-4 font-semibold text-gray-800">平均MAE</th>
                        <th className="text-center py-3 px-4 font-semibold text-gray-800">平均R²</th>
                        <th className="text-center py-3 px-4 font-semibold text-gray-800">RMSE标准差</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(modelResults).map(([modelName, { cvResults }]) => (
                        <tr key={modelName} className="border-b hover:bg-gray-50">
                          <td className="py-3 px-4 font-medium">{modelName}</td>
                          <td className="py-3 px-4 text-center">{cvResults.averageRMSE.toFixed(2)}</td>
                          <td className="py-3 px-4 text-center">{cvResults.averageMAE.toFixed(2)}</td>
                          <td className="py-3 px-4 text-center">{cvResults.averageR2.toFixed(3)}</td>
                          <td className="py-3 px-4 text-center">{cvResults.standardDeviationRMSE.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* 选中模型的详细图表 */}
                <ModelPerformanceChart
                  cvResults={modelResults[selectedModel].cvResults}
                  modelName={selectedModel}
                />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;