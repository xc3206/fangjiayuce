# 房价预测系统

这是一个完整的房价预测机器学习系统，实现了从数据生成到模型评估的完整流程。

## 功能特点

### 🏠 数据生成
- 自动生成1000条模拟房屋数据
- 包含8个特征：面积、卧室数、浴室数、房龄、楼层、停车位、花园、距离市中心
- 基于真实房价影响因素的价格计算模型

### 🤖 机器学习模型
实现了6种不同的回归模型：
- **线性回归** - 基础线性模型
- **岭回归** - 带L2正则化的线性模型
- **Lasso回归** - 带L1正则化的线性模型
- **随机森林** - 集成学习模型
- **梯度提升** - 梯度提升决策树
- **支持向量机** - 非线性回归模型

### 📊 模型评估
- **五折交叉验证** - 确保模型性能的可靠性
- **多种评估指标** - RMSE、MAE、R²
- **统计分析** - 均值、标准差、置信区间

### 📈 可视化分析
- 数据分布分析图
- 特征相关性热力图
- 模型性能对比图
- 预测值vs实际值散点图
- 残差分析图
- 特征重要性图
- 交叉验证分布箱线图

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行系统

```bash
python main.py
```

## 系统架构

```
房价预测系统/
├── main.py              # 主程序入口
├── data_generator.py    # 数据生成模块
├── models.py           # 机器学习模型模块
├── visualization.py    # 可视化模块
├── requirements.txt    # 依赖包列表
└── README.md          # 说明文档
```

## 核心模块说明

### 1. 数据生成器 (data_generator.py)
- `HouseDataGenerator` 类负责生成模拟房屋数据
- 基于真实房价影响因素设计价格计算公式
- 支持自定义数据量和随机种子

### 2. 模型预测器 (models.py)
- `HousePricePredictor` 类集成多种机器学习模型
- 自动处理数据标准化
- 支持交叉验证和批量预测
- 提供特征重要性分析

### 3. 可视化工具 (visualization.py)
- `ModelVisualizer` 类提供丰富的图表功能
- 支持中文显示
- 自动布局和美化
- 多种图表类型

## 使用示例

### 基本使用
```python
from data_generator import HouseDataGenerator
from models import HousePricePredictor
from visualization import ModelVisualizer

# 生成数据
generator = HouseDataGenerator()
data = generator.generate_house_data(1000)
X, y = generator.get_feature_target_split(data)

# 训练模型
predictor = HousePricePredictor()
predictor.train_models(X, y)

# 交叉验证
cv_results = predictor.cross_validate_models(X, y)

# 可视化
visualizer = ModelVisualizer()
visualizer.plot_cv_results(cv_results)
```

### 预测新房屋价格
```python
import pandas as pd

# 新房屋数据
new_house = pd.DataFrame({
    'area': [120.0],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [5.0],
    'floor': [8],
    'has_parking': [1],
    'has_garden': [1],
    'distance_to_center': [8.5]
})

# 预测价格
predictions = predictor.predict(new_house)
print(f"预测价格: {predictions['随机森林'][0]:.1f} 万元")
```

## 性能指标说明

- **RMSE (Root Mean Square Error)**: 均方根误差，越小越好
- **MAE (Mean Absolute Error)**: 平均绝对误差，越小越好
- **R² (R-squared)**: 决定系数，越接近1越好

## 扩展建议

1. **数据增强**: 添加更多特征如学区、交通便利性等
2. **模型优化**: 调整超参数，尝试深度学习模型
3. **真实数据**: 使用真实房价数据替换模拟数据
4. **在线预测**: 开发Web界面提供在线预测服务
5. **时间序列**: 考虑房价的时间变化趋势

## 技术栈

- **Python 3.7+**
- **scikit-learn**: 机器学习算法
- **pandas**: 数据处理
- **numpy**: 数值计算
- **matplotlib**: 基础绘图
- **seaborn**: 统计图表

## 许可证

MIT License