import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelVisualizer:
    """模型性能可视化工具"""
    
    def __init__(self, figsize: tuple = (15, 10)):
        self.figsize = figsize
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_data_distribution(self, data: pd.DataFrame):
        """
        绘制数据分布图
        
        Args:
            data: 房屋数据DataFrame
        """
        fig, axes = plt.subplots(3, 3, figsize=self.figsize)
        fig.suptitle('房屋数据分布分析', fontsize=16, fontweight='bold')
        
        # 数值特征分布
        numeric_features = ['area', 'age', 'distance_to_center', 'price']
        for i, feature in enumerate(numeric_features):
            row, col = i // 3, i % 3
            axes[row, col].hist(data[feature], bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{feature} 分布')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('频次')
        
        # 分类特征分布
        categorical_features = ['bedrooms', 'bathrooms', 'floor', 'has_parking', 'has_garden']
        for i, feature in enumerate(categorical_features, 4):
            row, col = i // 3, i % 3
            data[feature].value_counts().plot(kind='bar', ax=axes[row, col], alpha=0.7)
            axes[row, col].set_title(f'{feature} 分布')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('频次')
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame):
        """
        绘制特征相关性矩阵
        
        Args:
            data: 房屋数据DataFrame
        """
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('特征相关性矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_cv_results(self, cv_results: Dict):
        """
        绘制交叉验证结果对比图
        
        Args:
            cv_results: 交叉验证结果字典
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        fig.suptitle('模型交叉验证性能对比', fontsize=16, fontweight='bold')
        
        models = list(cv_results.keys())
        
        # RMSE对比
        rmse_means = [cv_results[model]['rmse_mean'] for model in models]
        rmse_stds = [cv_results[model]['rmse_std'] for model in models]
        
        axes[0].bar(models, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
        axes[0].set_title('RMSE (越小越好)')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE对比
        mae_means = [cv_results[model]['mae_mean'] for model in models]
        mae_stds = [cv_results[model]['mae_std'] for model in models]
        
        axes[1].bar(models, mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
        axes[1].set_title('MAE (越小越好)')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R²对比
        r2_means = [cv_results[model]['r2_mean'] for model in models]
        r2_stds = [cv_results[model]['r2_std'] for model in models]
        
        axes[2].bar(models, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
        axes[2].set_title('R² (越大越好)')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]):
        """
        绘制预测值vs实际值散点图
        
        Args:
            y_true: 真实值
            predictions: 各模型预测值字典
        """
        n_models = len(predictions)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('预测值 vs 实际值对比', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # 散点图
            ax.scatter(y_true, y_pred, alpha=0.6, s=30)
            
            # 完美预测线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
            
            # 计算R²
            r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
            ax.set_title(f'{model_name} (R² = {r2:.3f})')
            ax.set_xlabel('实际价格')
            ax.set_ylabel('预测价格')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]):
        """
        绘制残差图
        
        Args:
            y_true: 真实值
            predictions: 各模型预测值字典
        """
        n_models = len(predictions)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('残差分析', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            residuals = y_true - y_pred
            
            # 残差散点图
            ax.scatter(y_pred, residuals, alpha=0.6, s=30)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            
            ax.set_title(f'{model_name} 残差图')
            ax.set_xlabel('预测价格')
            ax.set_ylabel('残差 (实际值 - 预测值)')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.Series, model_name: str):
        """
        绘制特征重要性图
        
        Args:
            feature_importance: 特征重要性Series
            model_name: 模型名称
        """
        plt.figure(figsize=(10, 6))
        
        # 特征重要性条形图
        bars = plt.bar(range(len(feature_importance)), feature_importance.values, alpha=0.7)
        plt.title(f'{model_name} - 特征重要性', fontsize=14, fontweight='bold')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.xticks(range(len(feature_importance)), feature_importance.index, rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, feature_importance.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, cv_results: Dict):
        """
        绘制学习曲线（各折的性能变化）
        
        Args:
            cv_results: 交叉验证结果字典
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        fig.suptitle('交叉验证各折性能分布', fontsize=16, fontweight='bold')
        
        models = list(cv_results.keys())
        
        # RMSE分布
        rmse_data = [cv_results[model]['rmse_scores'] for model in models]
        axes[0].boxplot(rmse_data, labels=models)
        axes[0].set_title('RMSE分布')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE分布
        mae_data = [cv_results[model]['mae_scores'] for model in models]
        axes[1].boxplot(mae_data, labels=models)
        axes[1].set_title('MAE分布')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R²分布
        r2_data = [cv_results[model]['r2_scores'] for model in models]
        axes[2].boxplot(r2_data, labels=models)
        axes[2].set_title('R²分布')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()