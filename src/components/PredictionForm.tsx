import React, { useState } from 'react';
import { Calculator, Home, MapPin } from 'lucide-react';

export interface PredictionInput {
  area: number;
  bedrooms: number;
  bathrooms: number;
  age: number;
  floor: number;
  hasParking: boolean;
  hasGarden: boolean;
  distanceToCenter: number;
}

interface PredictionFormProps {
  onPredict: (input: PredictionInput) => void;
  predictions: { [key: string]: number };
}

export const PredictionForm: React.FC<PredictionFormProps> = ({ onPredict, predictions }) => {
  const [formData, setFormData] = useState<PredictionInput>({
    area: 100,
    bedrooms: 2,
    bathrooms: 1,
    age: 5,
    floor: 3,
    hasParking: true,
    hasGarden: false,
    distanceToCenter: 8
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(formData);
  };

  const handleInputChange = (field: keyof PredictionInput, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="flex items-center mb-6">
        <Calculator className="w-6 h-6 text-blue-600 mr-2" />
        <h2 className="text-xl font-bold text-gray-800">房价预测</h2>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Home className="w-4 h-4 inline mr-1" />
              面积 (平方米)
            </label>
            <input
              type="number"
              value={formData.area}
              onChange={(e) => handleInputChange('area', parseFloat(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              min="30"
              max="300"
              step="0.1"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              卧室数量
            </label>
            <input
              type="number"
              value={formData.bedrooms}
              onChange={(e) => handleInputChange('bedrooms', parseInt(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              min="1"
              max="6"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              浴室数量
            </label>
            <input
              type="number"
              value={formData.bathrooms}
              onChange={(e) => handleInputChange('bathrooms', parseInt(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              min="1"
              max="4"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              房龄 (年)
            </label>
            <input
              type="number"
              value={formData.age}
              onChange={(e) => handleInputChange('age', parseFloat(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              min="0"
              max="50"
              step="0.1"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              楼层
            </label>
            <input
              type="number"
              value={formData.floor}
              onChange={(e) => handleInputChange('floor', parseInt(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              min="1"
              max="30"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <MapPin className="w-4 h-4 inline mr-1" />
              距离市中心 (公里)
            </label>
            <input
              type="number"
              value={formData.distanceToCenter}
              onChange={(e) => handleInputChange('distanceToCenter', parseFloat(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              min="1"
              max="50"
              step="0.1"
            />
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <label className="flex items-center space-x-2 p-3 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50">
            <input
              type="checkbox"
              checked={formData.hasParking}
              onChange={(e) => handleInputChange('hasParking', e.target.checked)}
              className="w-4 h-4 text-blue-600"
            />
            <span className="text-sm font-medium text-gray-700">有停车位</span>
          </label>
          
          <label className="flex items-center space-x-2 p-3 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50">
            <input
              type="checkbox"
              checked={formData.hasGarden}
              onChange={(e) => handleInputChange('hasGarden', e.target.checked)}
              className="w-4 h-4 text-blue-600"
            />
            <span className="text-sm font-medium text-gray-700">有花园</span>
          </label>
        </div>
        
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          预测房价
        </button>
      </form>
      
      {Object.keys(predictions).length > 0 && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-3">预测结果</h3>
          <div className="space-y-2">
            {Object.entries(predictions).map(([model, price]) => (
              <div key={model} className="flex justify-between items-center">
                <span className="text-gray-700">{model}:</span>
                <span className="font-bold text-green-600">
                  {price.toFixed(1)} 万元
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};