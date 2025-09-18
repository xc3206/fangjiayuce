import numpy as np
import pandas as pd
from typing import Tuple

class HouseDataGenerator:
    """房屋数据生成器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_house_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        生成模拟房屋数据
        
        Args:
            n_samples: 生成样本数量
            
        Returns:
            包含房屋特征和价格的DataFrame
        """
        # 基础特征生成
        area = np.random.uniform(50, 200, n_samples)  # 面积 50-200平方米
        bedrooms = np.random.randint(1, 5, n_samples)  # 卧室数 1-4个
        bathrooms = np.random.randint(1, 4, n_samples)  # 浴室数 1-3个
        age = np.random.uniform(0, 30, n_samples)  # 房龄 0-30年
        floor = np.random.randint(1, 21, n_samples)  # 楼层 1-20层
        has_parking = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # 60%有停车位
        has_garden = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30%有花园
        distance_to_center = np.random.uniform(2, 27, n_samples)  # 距离市中心 2-27公里
        
        # 根据特征计算价格（带噪声）
        price = (area * 0.8 +  # 面积影响
                bedrooms * 15 +  # 卧室数影响
                bathrooms * 10 +  # 浴室数影响
                (30 - age) * 0.5 +  # 房龄影响（越新越贵）
                has_parking * 20 +  # 停车位加价
                has_garden * 25 +  # 花园加价
                np.maximum(0, (15 - distance_to_center)) * 2)  # 距离市中心越近越贵
        
        # 添加随机噪声
        noise = np.random.normal(0, 20, n_samples)
        price += noise
        price = np.maximum(30, price)  # 最低30万
        
        # 创建DataFrame
        data = pd.DataFrame({
            'area': np.round(area, 1),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': np.round(age, 1),
            'floor': floor,
            'has_parking': has_parking,
            'has_garden': has_garden,
            'distance_to_center': np.round(distance_to_center, 1),
            'price': np.round(price, 1)
        })
        
        return data
    
    def get_feature_target_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        分离特征和目标变量
        
        Args:
            data: 完整数据集
            
        Returns:
            特征矩阵X和目标向量y
        """
        feature_columns = ['area', 'bedrooms', 'bathrooms', 'age', 'floor', 
                          'has_parking', 'has_garden', 'distance_to_center']
        X = data[feature_columns]
        y = data['price']
        
        return X, y