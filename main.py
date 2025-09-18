#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房价预测系统主程序
实现了完整的机器学习流程：数据生成、模型训练、交叉验证、性能评估和可视化
"""

import pandas as pd
import numpy as np
from data_generator import HouseDataGenerator
from models import HousePricePredictor
from visualization import ModelVisualizer
import warnings
warnings.filterwarnings('ignore')

def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def print_cv_results_table(cv_results: dict):
    """打印交叉验证结果表格"""
    print("\n交叉验证结果汇总:")
    print("-" * 80)
    print(f"{'模型名称':<12} {'RMSE均值':<10} {'RMSE标准差':<12} {'MAE均值':<10} {'MAE标准差':<12} {'R²均值':<8} {'R²标准差':<10}")
    print("-" * 80)
    
    for model_name, results in cv_results.items():
        print(f"{model_name:<12} {results['rmse_mean']:<10.2f} {results['rmse_std']:<12.2f} "
              f"{results['mae_mean']:<10.2f} {results['mae_std']:<12.2f} "
              f"{results['r2_mean']:<8.3f} {results['r2_std']:<10.3f}")
    print("-" * 80)

def demonstrate_prediction(predictor: HousePricePredictor, feature_names: list):
    """演示预测功能"""
    print_separator("预测演示")
    
    # 创建示例房屋数据
    sample_houses = pd.DataFrame({
        'area': [120.0, 80.0, 150.0],
        'bedrooms': [3, 2, 4],
        'bathrooms': [2, 1, 3],
        'age': [5.0, 15.0, 2.0],
        'floor': [8, 3, 12],
        'has_parking': [1, 0, 1],
        'has_garden': [1, 0, 1],
        'distance_to_center': [8.5, 15.2, 5.0]
    })
    
    print("示例房屋信息:")
    for i, (_, house) in enumerate(sample_houses.iterrows(), 1):
        print(f"\n房屋 {i}:")
        print(f"  面积: {house['area']}平方米")
        print(f"  卧室: {house['bedrooms']}个")
        print(f"  浴室: {house['bathrooms']}个")
        print(f"  房龄: {house['age']}年")
        print(f"  楼层: {house['floor']}层")
        print(f"  停车位: {'有' if house['has_parking'] else '无'}")
        print(f"  花园: {'有' if house['has_garden'] else '无'}")
        print(f"  距离市中心: {house['distance_to_center']}公里")
    
    # 进行预测
    predictions = predictor.predict(sample_houses)
    
    print("\n预测结果:")
    print("-" * 60)
    print(f"{'房屋':<6} {'线性回归':<10} {'岭回归':<10} {'Lasso回归':<12} {'随机森林':<10} {'梯度提升':<10} {'支持向量机':<12}")
    print("-" * 60)
    
    for i in range(len(sample_houses)):
        print(f"房屋{i+1:<5}", end="")
        for model_name in predictions.keys():
            price = predictions[model_name][i]
            print(f"{price:<10.1f}", end=" ")
        print()
    print("-" * 60)

def main():
    """主函数"""
    print_separator("房价预测系统")
    print("本系统实现了完整的机器学习流程，包括：")
    print("1. 模拟房屋数据生成")
    print("2. 多种机器学习模型训练")
    print("3. 五折交叉验证性能评估")
    print("4. 可视化分析和预测演示")
    
    # 1. 生成数据
    print_separator("数据生成")
    generator = HouseDataGenerator(random_state=42)
    data = generator.generate_house_data(n_samples=1000)
    X, y = generator.get_feature_target_split(data)
    
    print(f"生成了 {len(data)} 条房屋数据")
    print(f"特征数量: {X.shape[1]}")
    print(f"特征名称: {list(X.columns)}")
    print(f"\n数据基本统计:")
    print(data.describe())
    
    # 2. 数据可视化
    print_separator("数据分析可视化")
    visualizer = ModelVisualizer()
    
    print("绘制数据分布图...")
    visualizer.plot_data_distribution(data)
    
    print("绘制特征相关性矩阵...")
    visualizer.plot_correlation_matrix(data)
    
    # 3. 模型训练
    print_separator("模型训练")
    predictor = HousePricePredictor(random_state=42)
    predictor.train_models(X, y)
    
    # 4. 交叉验证
    print_separator("五折交叉验证")
    cv_results = predictor.cross_validate_models(X, y, cv_folds=5)
    
    # 打印结果表格
    print_cv_results_table(cv_results)
    
    # 找出最佳模型
    best_rmse_model = predictor.get_best_model('rmse')
    best_r2_model = predictor.get_best_model('r2')
    
    print(f"\n最佳模型 (RMSE): {best_rmse_model} (RMSE = {cv_results[best_rmse_model]['rmse_mean']:.2f})")
    print(f"最佳模型 (R²): {best_r2_model} (R² = {cv_results[best_r2_model]['r2_mean']:.3f})")
    
    # 5. 性能可视化
    print_separator("性能可视化")
    
    print("绘制交叉验证结果对比图...")
    visualizer.plot_cv_results(cv_results)
    
    print("绘制交叉验证分布箱线图...")
    visualizer.plot_learning_curves(cv_results)
    
    # 6. 预测效果可视化
    print_separator("预测效果分析")
    
    # 使用训练好的模型对训练数据进行预测（用于可视化）
    train_predictions = predictor.predict(X)
    
    print("绘制预测值vs实际值散点图...")
    visualizer.plot_prediction_vs_actual(y.values, train_predictions)
    
    print("绘制残差分析图...")
    visualizer.plot_residuals(y.values, train_predictions)
    
    # 7. 特征重要性分析
    print_separator("特征重要性分析")
    
    # 分析随机森林的特征重要性
    try:
        rf_importance = predictor.get_feature_importance('随机森林')
        print("随机森林特征重要性排序:")
        for feature, importance in rf_importance.items():
            print(f"  {feature}: {importance:.3f}")
        
        print("绘制特征重要性图...")
        visualizer.plot_feature_importance(rf_importance, '随机森林')
    except Exception as e:
        print(f"特征重要性分析出错: {e}")
    
    # 8. 预测演示
    demonstrate_prediction(predictor, list(X.columns))
    
    # 9. 总结
    print_separator("系统总结")
    print("房价预测系统运行完成！")
    print(f"✓ 成功生成 {len(data)} 条模拟房屋数据")
    print(f"✓ 训练了 {len(predictor.models)} 个机器学习模型")
    print(f"✓ 完成了五折交叉验证性能评估")
    print(f"✓ 生成了多种可视化图表")
    print(f"✓ 最佳模型: {best_r2_model} (R² = {cv_results[best_r2_model]['r2_mean']:.3f})")
    
    print("\n建议:")
    print("1. 可以调整模型参数以获得更好的性能")
    print("2. 可以增加更多特征来提高预测准确性")
    print("3. 可以收集真实数据来替换模拟数据")
    print("4. 可以尝试集成学习方法来进一步提升性能")

if __name__ == "__main__":
    main()