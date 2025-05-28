# Tesla-Stock-price-perdition

# Tesla 股票预测应用

这是一个使用 Streamlit 构建的 Tesla 股票预测应用，集成了 LSTM、EGARCH 和 LightGBM 三种模型。

## 功能特点

- 支持上传 Tesla 股票 CSV 数据
- 提供基础设置和高级设置选项
- 支持多种预测模型的可视化比较
- 提供模型权重的手动和自动优化
- 包含完整的预测评估指标

## 使用方法

1. 运行应用：
```bash
streamlit run app.py
```

2. 数据要求：
   - 上传的 CSV 文件必须包含以下列：
     - Date
     - Open
     - High
     - Low
     - Close
     - Volume

3. 参数设置：
   - 基础设置：
     - Forecast Days (5-60天)
     - LSTM Window Size (5-60天)
   - 高级设置：
     - EGARCH History Length (100-200天)
     - LightGBM Feature Window (20-60天)

4. 预测模型：
   - 可选择显示：
     - Actual
     - LSTM
     - EGARCH
     - LightGBM
     - Ensemble(user)
     - Ensemble(opt)

5. 权重模式：
   - Optimal (locked)：使用优化权重
   - Manual：手动调整权重

## 评估指标

- RMSE
- MAE
- MAPE
- R2
- Direction Accuracy

## 依赖包

- streamlit
- pandas
- numpy
- tensorflow
- arch
- lightgbm
- scikit-learn
