import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Scatter, Bar } from 'react-chartjs-2';
import { ValidationResult, CrossValidationResult } from '../utils/crossValidation';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface ModelPerformanceChartProps {
  cvResults: CrossValidationResult;
  modelName: string;
}

export const ModelPerformanceChart: React.FC<ModelPerformanceChartProps> = ({
  cvResults,
  modelName
}) => {
  // 合并所有折的预测结果
  const allPredictions = cvResults.foldResults.flatMap(result => result.predictions);
  const allActual = cvResults.foldResults.flatMap(result => result.actual);

  // 实际值 vs 预测值散点图数据
  const scatterData = {
    datasets: [
      {
        label: '预测值 vs 实际值',
        data: allActual.map((actual, index) => ({
          x: actual,
          y: allPredictions[index]
        })),
        backgroundColor: 'rgba(59, 130, 246, 0.6)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
      },
      {
        label: '完美预测线',
        data: [
          { x: Math.min(...allActual), y: Math.min(...allActual) },
          { x: Math.max(...allActual), y: Math.max(...allActual) }
        ],
        backgroundColor: 'rgba(239, 68, 68, 0.8)',
        borderColor: 'rgba(239, 68, 68, 1)',
        borderWidth: 2,
        showLine: true,
        pointRadius: 0,
      }
    ]
  };

  // 交叉验证分数柱状图数据
  const barData = {
    labels: cvResults.foldResults.map((_, index) => `第${index + 1}折`),
    datasets: [
      {
        label: 'RMSE',
        data: cvResults.foldResults.map(result => result.rmse),
        backgroundColor: 'rgba(34, 197, 94, 0.6)',
        borderColor: 'rgba(34, 197, 94, 1)',
        borderWidth: 1,
      },
      {
        label: 'MAE',
        data: cvResults.foldResults.map(result => result.mae),
        backgroundColor: 'rgba(251, 146, 60, 0.6)',
        borderColor: 'rgba(251, 146, 60, 1)',
        borderWidth: 1,
      }
    ]
  };

  const scatterOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${modelName} - 实际值 vs 预测值`,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: '实际房价 (万元)'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: '预测房价 (万元)'
        }
      }
    }
  };

  const barOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${modelName} - 交叉验证误差分布`,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: '误差值'
        }
      }
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg text-center">
          <h3 className="font-semibold text-blue-800">平均RMSE</h3>
          <p className="text-2xl font-bold text-blue-600">
            {cvResults.averageRMSE.toFixed(2)}
          </p>
          <p className="text-sm text-blue-500">
            ±{cvResults.standardDeviationRMSE.toFixed(2)}
          </p>
        </div>
        <div className="bg-green-50 p-4 rounded-lg text-center">
          <h3 className="font-semibold text-green-800">平均MAE</h3>
          <p className="text-2xl font-bold text-green-600">
            {cvResults.averageMAE.toFixed(2)}
          </p>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg text-center">
          <h3 className="font-semibold text-purple-800">平均R²</h3>
          <p className="text-2xl font-bold text-purple-600">
            {cvResults.averageR2.toFixed(3)}
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <Scatter data={scatterData} options={scatterOptions} />
        </div>
        <div>
          <Bar data={barData} options={barOptions} />
        </div>
      </div>
    </div>
  );
};