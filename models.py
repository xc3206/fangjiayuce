import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    """房价预测模型集合"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.cv_results = {}
        
        # 初始化模型
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化所有模型"""
        self.models = {
            '线性回归': LinearRegression(),
            '岭回归': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso回归': Lasso(alpha=1.0, random_state=self.random_state),
            '随机森林': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            '梯度提升': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            '支持向量机': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        # 为需要标准化的模型初始化缩放器
        scaling_models = ['岭回归', 'Lasso回归', '支持向量机']
        for model_name in scaling_models:
            self.scalers[model_name] = StandardScaler()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series):
        """
        训练所有模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
        """
        print("开始训练模型...")
        
        for name, model in self.models.items():
            print(f"训练 {name}...")
            
            if name in self.scalers:
                # 需要标准化的模型
                X_scaled = self.scalers[name].fit_transform(X)
                model.fit(X_scaled, y)
            else:
                # 不需要标准化的模型
                model.fit(X, y)
        
        print("所有模型训练完成！")
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict:
        """
        使用交叉验证评估所有模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
            cv_folds: 交叉验证折数
            
        Returns:
            包含所有模型交叉验证结果的字典
        """
        print(f"开始 {cv_folds} 折交叉验证...")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"交叉验证 {name}...")
            
            if name in self.scalers:
                # 对于需要标准化的模型，在每个fold中分别进行标准化
                rmse_scores = []
                mae_scores = []
                r2_scores = []
                
                for train_idx, val_idx in kfold.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # 标准化
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # 训练和预测
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                    
                    # 计算指标
                    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
                    mae_scores.append(mean_absolute_error(y_val, y_pred))
                    r2_scores.append(r2_score(y_val, y_pred))
                
                self.cv_results[name] = {
                    'rmse_mean': np.mean(rmse_scores),
                    'rmse_std': np.std(rmse_scores),
                    'mae_mean': np.mean(mae_scores),
                    'mae_std': np.std(mae_scores),
                    'r2_mean': np.mean(r2_scores),
                    'r2_std': np.std(r2_scores),
                    'rmse_scores': rmse_scores,
                    'mae_scores': mae_scores,
                    'r2_scores': r2_scores
                }
            else:
                # 对于不需要标准化的模型，直接使用sklearn的cross_val_score
                rmse_scores = np.sqrt(-cross_val_score(
                    model, X, y, cv=kfold, scoring='neg_mean_squared_error'
                ))
                mae_scores = -cross_val_score(
                    model, X, y, cv=kfold, scoring='neg_mean_absolute_error'
                )
                r2_scores = cross_val_score(
                    model, X, y, cv=kfold, scoring='r2'
                )
                
                self.cv_results[name] = {
                    'rmse_mean': np.mean(rmse_scores),
                    'rmse_std': np.std(rmse_scores),
                    'mae_mean': np.mean(mae_scores),
                    'mae_std': np.std(mae_scores),
                    'r2_mean': np.mean(r2_scores),
                    'r2_std': np.std(r2_scores),
                    'rmse_scores': rmse_scores,
                    'mae_scores': mae_scores,
                    'r2_scores': r2_scores
                }
        
        print("交叉验证完成！")
        return self.cv_results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> Dict[str, np.ndarray]:
        """
        使用训练好的模型进行预测
        
        Args:
            X: 特征矩阵
            model_name: 指定模型名称，如果为None则使用所有模型
            
        Returns:
            包含所有模型预测结果的字典
        """
        predictions = {}
        
        models_to_use = [model_name] if model_name else self.models.keys()
        
        for name in models_to_use:
            if name not in self.models:
                continue
                
            model = self.models[name]
            
            if name in self.scalers:
                X_scaled = self.scalers[name].transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        
        return predictions
    
    def get_feature_importance(self, model_name: str) -> pd.Series:
        """
        获取模型的特征重要性
        
        Args:
            model_name: 模型名称
            
        Returns:
            特征重要性Series
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        feature_names = ['area', 'bedrooms', 'bathrooms', 'age', 'floor', 
                        'has_parking', 'has_garden', 'distance_to_center']
        
        if hasattr(model, 'feature_importances_'):
            # 树模型有feature_importances_属性
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 线性模型有coef_属性
            importance = np.abs(model.coef_)
        else:
            raise ValueError(f"模型 {model_name} 不支持特征重要性分析")
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
    
    def get_best_model(self, metric: str = 'rmse') -> str:
        """
        根据指定指标获取最佳模型
        
        Args:
            metric: 评估指标 ('rmse', 'mae', 'r2')
            
        Returns:
            最佳模型名称
        """
        if not self.cv_results:
            raise ValueError("请先运行交叉验证")
        
        if metric == 'r2':
            # R2越大越好
            best_model = max(self.cv_results.keys(), 
                           key=lambda x: self.cv_results[x]['r2_mean'])
        else:
            # RMSE和MAE越小越好
            best_model = min(self.cv_results.keys(), 
                           key=lambda x: self.cv_results[x][f'{metric}_mean'])
        
        return best_model