import streamlit as st
import pandas as pd
import numpy as np
from config import MODEL_PATHS, DEFAULT_PARAMS, WEIGHT_CONSTRAINTS
from data_utils import load_data
from models_lstm import LSTMPredictor
from models_egarch import EGARCHPredictor
from models_lgb import LightGBMPredictor
from ensemble import EnsembleOptimizer
from metrics import calculate_metrics
from visualization import plot_forecasts, plot_weights

# 设置页面配置
st.set_page_config(page_title="Tesla Forecast: LSTM vs EGARCH vs LightGBM", layout="centered")

# 标题
st.title("Tesla Stock Forecast — LSTM vs EGARCH vs LightGBM")

# 文件上传
uploaded_file = st.file_uploader("Upload Tesla CSV", type="csv")

# 缓存刷新函数
def set_need_refresh():
    st.session_state.need_refresh = True

# 基础设置
forecast_days = st.slider("Forecast Days", 5, 60, DEFAULT_PARAMS["FORECAST_DAYS"], on_change=set_need_refresh)
window_size = st.slider("LSTM Window Size", 5, 60, DEFAULT_PARAMS["LSTM_WINDOW"], on_change=set_need_refresh)

# 高级设置
with st.sidebar.expander("Advanced Settings", expanded=False):
    egarch_history = st.slider("EGARCH History Length", 100, 200, DEFAULT_PARAMS["EGARCH_HISTORY"], on_change=set_need_refresh)
    lgb_history = st.slider("LightGBM Feature Window", 20, 60, DEFAULT_PARAMS["LGB_HISTORY"], on_change=set_need_refresh)

if uploaded_file:
    # 加载数据
    df = load_data(uploaded_file)
    
    # 初始化预测器
    lstm_predictor = LSTMPredictor(MODEL_PATHS["LSTM_MODEL"], MODEL_PATHS["LSTM_SCALER"])
    egarch_predictor = EGARCHPredictor(MODEL_PATHS["EGARCH_TEMPLATE"])
    lgb_predictor = LightGBMPredictor(MODEL_PATHS["LGB_MODEL"], MODEL_PATHS["LGB_SCALER"])
    
    # 执行预测
    with st.spinner("Generating predictions..."):
        # LSTM预测
        lstm_results = lstm_predictor.predict(df, window_size, forecast_days)

        # EGARCH预测
        egarch_results = egarch_predictor.predict(df, egarch_history, forecast_days)
        
        # LightGBM预测
        lgb_results = lgb_predictor.predict(df, lgb_history, forecast_days)
        
        # 准备预测结果字典
        preds_dict = {
            "LSTM": lstm_results,
            "EGARCH": egarch_results,
            "LightGBM": lgb_results
        }
        
        # 优化权重
        optimizer = EnsembleOptimizer(
            {k: v["predictions"] for k, v in preds_dict.items()},
            lgb_results["evaluation"]
        )
        
        if "opt_w" not in st.session_state or st.session_state.get("need_refresh", True):
            opt_w = optimizer.optimize()
            st.session_state.opt_w = opt_w
        opt_w = st.session_state.opt_w
        
        # 计算优化后的融合预测
        ensemble_opt = optimizer.blend(opt_w)
        preds_dict["Ensemble(opt)"] = {
            "history": lgb_results["history"],
            "predictions": ensemble_opt
        }
    
    # 权重模式切换
    mode = st.sidebar.radio("Weight Mode", ["Optimal (locked)", "Manual"], horizontal=True)
    
    if mode == "Manual":
        st.sidebar.write("⬇️ 调节权重滑块 ⬇️")
        weights = []
        for i, model in enumerate(["LSTM", "EGARCH", "LightGBM"]):
            weight = st.sidebar.slider(
                model, 0.0, 1.0,
                value=st.session_state.get(f"w_{model.lower()}", opt_w[i]),
                key=f"w_{model.lower()}",
                on_change=set_need_refresh
            )
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 计算手动融合预测
        ensemble_manual = optimizer.blend(weights)
        preds_dict["Ensemble(user)"] = {
            "history": lgb_results["history"],
            "predictions": ensemble_manual
        }
    
    # 选择要显示的线条
    lines_to_show = st.multiselect(
        "Select lines to show",
        ["Actual", "LSTM", "EGARCH", "LightGBM", "Ensemble(user)", "Ensemble(opt)"],
        default=["Actual", "LSTM", "EGARCH", "LightGBM", "Ensemble(user)", "Ensemble(opt)"]
    )
    
    # 绘制预测图
    fig = plot_forecasts(
        df["Date"],
        lgb_results["evaluation"],
        preds_dict,
        len(lgb_results["history"]),
        lines_to_show
    )
            st.pyplot(fig)

    # 显示权重饼图
    if mode == "Manual":
        fig2 = plot_weights(weights, ["LSTM", "EGARCH", "LGB"])
    else:
        fig2 = plot_weights(opt_w, ["LSTM", "EGARCH", "LGB"])
    st.sidebar.pyplot(fig2)
    
    # 计算并显示指标
    st.subheader("Forecast Accuracy Metrics")
    metrics = {}
    for model_name, data in preds_dict.items():
        metrics[model_name] = calculate_metrics(
            lgb_results["evaluation"],
            data["predictions"]
        )
    
    # 显示指标表格
    for model_name, stats in metrics.items():
        st.write(f"**{model_name}**: RMSE = {stats['RMSE']:.2f}, "
                f"MAE = {stats['MAE']:.2f}, "
                f"MAPE = {stats['MAPE']:.2f}%, "
                f"R2 = {stats['R2']:.2f}, "
                f"Direction Acc = {stats['Direction Acc']:.2f}%")
    
    # 添加保存功能
    if st.button("Save Results"):
        results_df = pd.DataFrame({
            "Date": df["Date"].iloc[-forecast_days:],
            "Actual": lgb_results["evaluation"]
        })
        
        for model_name, data in preds_dict.items():
            results_df[f"{model_name}_Pred"] = data["predictions"]
        
        results_df.to_csv("forecast_results.csv", index=False)
        st.success("Results saved to forecast_results.csv")
else:
    st.info("Please upload a Tesla stock CSV file to begin.")
