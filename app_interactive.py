import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from arch import arch_model
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import deque
from scipy.optimize import minimize, Bounds

st.set_page_config(page_title="Tesla Forecast: LSTM vs EGARCH vs LightGBM", layout="centered")

# Model and scaler files
EGARCH_TEMPLATE = "Chris_egarch_model.pkl"
LSTM_MODEL = "lstm_model.h5"
SCALER_FILE = "lstm_scaler.pkl"
LGB_MODEL = "lgb_model.pkl"
LGB_SCALER = "lgb_scaler.pkl"

@st.cache_resource(show_spinner=False)
def load_models():
    egarch_template = joblib.load(EGARCH_TEMPLATE)
    lstm_model = load_model(LSTM_MODEL, compile=False)
    lstm_scaler = joblib.load(SCALER_FILE)
    lgb_model = joblib.load(LGB_MODEL)
    lgb_scaler = joblib.load(LGB_SCALER)
    return egarch_template, lstm_model, lstm_scaler, lgb_model, lgb_scaler

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], 
                     usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    return df.dropna()

def mape(y_true, y_pred):
    # 确保输入是numpy数组
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # 避免除以零
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    
    # 计算MAPE
    mape_val = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(mape_val)  # 确保返回标量

def direction_accuracy(y_true, y_pred):
    # 确保输入是numpy数组
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # 计算方向准确率
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    accuracy = np.mean(true_direction == pred_direction) * 100
    return float(accuracy)  # 确保返回标量

def predict_lstm(model, data, steps, window):
    seq = deque(data[-window:].flatten(), maxlen=window)
    preds = []
    for _ in range(steps):
        x = np.array(seq).reshape(1, window, 1)
        p = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        seq.append(p)
    return np.array(preds)

def simulate_egarch(returns, days):
    am = arch_model(returns, vol="EGARCH", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    sim = am.simulate(res.params, nobs=days)["data"].values.flatten()
    return sim

# --- Streamlit UI ---
st.title("Tesla Stock Forecast — LSTM vs EGARCH vs LightGBM")
uploaded_file = st.file_uploader("Upload Tesla CSV", type="csv")

# --- 滑块变动时自动刷新预测缓存 ---
def set_need_refresh():
    st.session_state.need_refresh = True

# 基础设置
forecast_days = st.slider("Forecast Days", 5, 60, 30, on_change=set_need_refresh)
window_size = st.slider("LSTM Window Size", 5, 60, 20, on_change=set_need_refresh)

# 高级设置
with st.sidebar.expander("Advanced Settings", expanded=False):
    egarch_history = st.slider("EGARCH History Length", 100, 200, 120, on_change=set_need_refresh)
    lgb_history = st.slider("LightGBM Feature Window", 20, 60, 30, on_change=set_need_refresh)

if uploaded_file:
    df = load_data(uploaded_file)
    egarch_template, lstm_model, scaler, lgb_model, lgb_scaler = load_models()

    # ===== 1. LSTM 模型预测 =====
    df_lstm = df.copy()
    scaled = scaler.transform(df_lstm[["Close"]])
    lstm_preds_scaled = predict_lstm(lstm_model, scaled, forecast_days, window_size)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
    
    # LSTM 的历史数据和评估数据
    hist_lstm = df["Close"].iloc[-(window_size + forecast_days):-forecast_days].values
    eval_actual_lstm = df["Close"].iloc[-(window_size + forecast_days):].values[window_size:]
    lstm_overlay = np.concatenate([hist_lstm, lstm_preds])
    
    print(f"LSTM预测长度: {len(lstm_preds)}")

    # ===== 2. EGARCH 模型预测 =====
    df_egarch = df.copy()
    df_egarch["log_return"] = np.log(df_egarch["Close"] / df_egarch["Close"].shift(1))
    
    # EGARCH 使用独立的历史窗口
    egarch_start = -(forecast_days + 1)
    returns = df_egarch["log_return"].iloc[egarch_start - egarch_history:egarch_start].dropna()
    last_price = df_egarch["Close"].iloc[egarch_start]
    
    def simulate_egarch_fixed(returns, days, seed=42):
        np.random.seed(seed)
        am = arch_model(returns, vol="EGARCH", p=1, q=1, mean="Zero", dist="normal")
        res = am.fit(disp="off")
        sim = am.simulate(res.params, nobs=days)["data"].values.flatten()
        return sim
    
    sim = simulate_egarch_fixed(returns, forecast_days, seed=42)
    egarch_preds = last_price * np.exp(np.cumsum(sim))
    
    # EGARCH 的历史数据和评估数据
    hist_egarch = df["Close"].iloc[egarch_start - 30:egarch_start].values  # 显示用30天
    eval_actual_egarch = df["Close"].iloc[egarch_start:egarch_start + forecast_days].values
    
    print(f"EGARCH预测长度: {len(egarch_preds)}")
    print(f"EGARCH评估数据长度: {len(eval_actual_egarch)}")
    
    egarch_overlay = np.concatenate([hist_egarch, egarch_preds])

    # ===== 3. LightGBM 模型预测 =====
    df_lgb = df.copy()
    df_lgb["Date"] = pd.to_datetime(df_lgb["Date"], errors="coerce")
    
    # LightGBM 使用独立的历史窗口
    lgb_start = -(forecast_days + lgb_history)
    real_close = df_lgb["Close"].iloc[lgb_start:-forecast_days].tolist()
    real_open = df_lgb["Open"].iloc[lgb_start:-forecast_days].tolist()
    real_high = df_lgb["High"].iloc[lgb_start:-forecast_days].tolist()
    real_low = df_lgb["Low"].iloc[lgb_start:-forecast_days].tolist()
    real_volume = df_lgb["Volume"].iloc[lgb_start:-forecast_days].tolist()
    real_dates = df_lgb["Date"].iloc[lgb_start:-forecast_days].tolist()
    
    print(f"LightGBM历史数据长度: {len(real_close)}")
    
    lgb_forecast = []
    current_date = real_dates[-1]
    
    for _ in range(forecast_days):
        # 使用真实历史数据构建特征
        lag_values = real_close[-5:]  # 使用真实值而不是预测值
        ma5_val = np.mean(lag_values)
        rsi_val = 100 - 100 / (1 + pd.Series(real_close).pct_change().rolling(14).mean().iloc[-1])
        
        # 构建特征行
        row = pd.DataFrame({
            "Open": [real_open[-1]],
            "High": [real_high[-1]],
            "Low": [real_low[-1]],
            "Volume": [real_volume[-1]],
            "Close_lag1": [lag_values[-1]],
            "Close_lag2": [lag_values[-2]],
            "Close_lag3": [lag_values[-3]],
            "Close_lag4": [lag_values[-4]],
            "Close_lag5": [lag_values[-5]],
            "MA5": [ma5_val],
            "RSI": [rsi_val],
            "Year": [current_date.year],
            "Month": [current_date.month],
            "DayOfWeek": [current_date.dayofweek],
            "Quarter": [current_date.quarter]
        })
        
        # 预测下一个值
        X_scaled = lgb_scaler.transform(row)
        next_price = lgb_model.predict(X_scaled)[0]
        lgb_forecast.append(next_price)
        
        # 更新日期和真实值列表
        current_date = current_date + pd.Timedelta(days=1)
        real_close.append(next_price)
        real_open.append(next_price)
        real_high.append(next_price)
        real_low.append(next_price)
        real_volume.append(real_volume[-1])
    
    # LightGBM 的历史数据和评估数据
    hist_lgb = df["Close"].iloc[lgb_start:-forecast_days].values
    eval_actual_lgb = df["Close"].iloc[-forecast_days:].values
    
    print(f"LightGBM预测长度: {len(lgb_forecast)}")
    print(f"LightGBM评估数据长度: {len(eval_actual_lgb)}")
    
    lgb_overlay = np.concatenate([hist_lgb, lgb_forecast])

    # ===== 4. 预测缓存 =====
    if "pred_cache" not in st.session_state or st.session_state.get("need_refresh", True):
        # 确保所有预测长度一致
        min_pred_len = min(len(lstm_preds), len(egarch_preds), len(lgb_forecast))
        print(f"最小预测长度: {min_pred_len}")
        
        if min_pred_len == 0:
            st.error(f"预测结果为空！\nLSTM: {len(lstm_preds)}\nEGARCH: {len(egarch_preds)}\nLightGBM: {len(lgb_forecast)}")
            st.stop()
            
        lstm_preds = lstm_preds[:min_pred_len]
        egarch_preds = egarch_preds[:min_pred_len]
        lgb_forecast = lgb_forecast[:min_pred_len]
        
        # 确保所有评估数据长度一致
        min_eval_len = min(len(eval_actual_lstm), len(eval_actual_egarch), len(eval_actual_lgb))
        print(f"最小评估数据长度: {min_eval_len}")
        
        if min_eval_len == 0:
            st.error(f"评估数据为空！\nLSTM: {len(eval_actual_lstm)}\nEGARCH: {len(eval_actual_egarch)}\nLightGBM: {len(eval_actual_lgb)}")
            st.stop()
            
        eval_actual_lstm = eval_actual_lstm[:min_eval_len]
        eval_actual_egarch = eval_actual_egarch[:min_eval_len]
        eval_actual_lgb = eval_actual_lgb[:min_eval_len]
        
        st.session_state.pred_cache = {
            "hist_lstm": hist_lstm,
            "hist_egarch": hist_egarch,
            "hist_lgb": hist_lgb,
            "dates": df["Date"].iloc[-max(len(hist_lstm), len(hist_egarch), len(hist_lgb)):],
            "eval_actual_lstm": eval_actual_lstm,
            "eval_actual_egarch": eval_actual_egarch,
            "eval_actual_lgb": eval_actual_lgb,
            "lstm_pred": lstm_preds,
            "eg_pred": egarch_preds,
            "lgb_pred": lgb_forecast
        }
        st.session_state.need_refresh = False
    p = st.session_state.pred_cache

    # ===== 5. 优化权重计算 =====
    if "opt_w" not in st.session_state or st.session_state.get("need_refresh", True):
        def obj(w, y, preds, lam=8, alpha=2.0):
            if len(y) == 0 or len(preds) == 0:
                return float('inf')  # 如果数据为空，返回无穷大
            blend = np.dot(w, preds)
            rmse = np.sqrt(((y-blend)**2).mean())
            # 添加保护措施，确保数组非空
            if len(y) > 1 and len(blend) > 1:
                hit = (np.sign(np.diff(y,prepend=y[0])) == np.sign(np.diff(blend,prepend=blend[0]))).mean()
            else:
                hit = 0.5  # 如果数据点太少，使用中性值
            reg = alpha * np.sum((w - 1/3)**2)
            return rmse + lam*(1-hit) + reg

        # 确保所有预测长度一致且非空
        min_len = min(len(p["lstm_pred"]), len(p["eg_pred"]), len(p["lgb_pred"]))
        if min_len == 0:
            st.error("预测结果为空，请检查数据")
            st.stop()
            
        P = np.vstack([
            p["lstm_pred"][:min_len],
            p["eg_pred"][:min_len],
            p["lgb_pred"][:min_len]
        ])
        y = p["eval_actual_lgb"][:min_len]  # 使用LightGBM的评估数据作为基准
        
        if len(y) == 0:
            st.error("评估数据为空，请检查数据")
            st.stop()
            
        cons = ({'type':'eq', 'fun': lambda w: w.sum()-1})
        bds = Bounds([0.15, 0.15, 0.15], [0.5, 0.5, 0.5])
        res = minimize(obj, x0=[0.34, 0.33, 0.33],
                      args=(y, P, 8, 2.0),
                      bounds=bds, constraints=cons)
        st.session_state.opt_w = res.x / res.x.sum()
    opt_w = st.session_state.opt_w
    
    # 确保所有预测长度一致且非空
    min_len = min(len(p["lstm_pred"]), len(p["eg_pred"]), len(p["lgb_pred"]))
    if min_len == 0:
        st.error("预测结果为空，请检查数据")
        st.stop()
        
    ensemble_opt = opt_w @ np.vstack([
        p["lstm_pred"][:min_len],
        p["eg_pred"][:min_len],
        p["lgb_pred"][:min_len]
    ])
    ensemble_opt_overlay = np.concatenate([p["hist_lgb"], ensemble_opt])

    # ===== 6. 权重模式切换 =====
    if "w_lstm" not in st.session_state or st.session_state.get("need_refresh", True):
        st.session_state.w_lstm = float(opt_w[0])
        st.session_state.w_eg = float(opt_w[1])
        st.session_state.w_lgb = float(opt_w[2])
    mode = st.sidebar.radio("Weight Mode", ["Optimal (locked)", "Manual"], horizontal=True)

    def adjust_weights(changed):
        keys = ["w_lstm", "w_eg", "w_lgb"]
        idx = keys.index(changed)
        w = np.array([st.session_state[k] for k in keys], float)
        diff = w.sum() - 1
        if abs(diff) < 1e-6: return
        others = [i for i in range(3) if i != idx]
        rest = w[others].sum()
        for j in others:
            w[j] = max(0, w[j] - diff * (w[j]/rest) if rest > 0 else 0)
        for k,val in zip(keys,w):
            st.session_state[k] = round(val,4)

    if mode == "Manual":
        st.sidebar.write("⬇️ 调节权重滑块 ⬇️")
        for k,label in zip(["w_lstm","w_eg","w_lgb"],["LSTM","EGARCH","LGB"]):
            st.sidebar.slider(label, 0.0, 1.0, key=k, on_change=adjust_weights, args=(k,))

    # ===== 7. Ensemble(user) 计算 =====
    weights = np.array([st.session_state.w_lstm,
                       st.session_state.w_eg,
                       st.session_state.w_lgb])
    # 确保所有预测长度一致且非空
    min_len = min(len(p["lstm_pred"]), len(p["eg_pred"]), len(p["lgb_pred"]))
    if min_len == 0:
        st.error("预测结果为空，请检查数据")
        st.stop()
        
    blend_pred = weights @ np.vstack([
        p["lstm_pred"][:min_len],
        p["eg_pred"][:min_len],
        p["lgb_pred"][:min_len]
    ])
    ensemble_overlay = np.concatenate([p["hist_lgb"], blend_pred])

    # ===== 8. 指标计算 =====
    def calc_metrics(y_true, y_pred):
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0, 0, 0
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape_val = mape(y_true, y_pred)
        hit = direction_accuracy(y_true, y_pred)
        return rmse, mape_val, hit

    rmse, mape_val, hit = calc_metrics(p["eval_actual_lgb"], blend_pred)
    st.sidebar.markdown("### 📊 Current Metrics")
    st.sidebar.metric("RMSE", f"{rmse:.2f}")
    st.sidebar.metric("MAPE", f"{mape_val:.2f}%")
    st.sidebar.metric("Hit-Ratio", f"{hit:.1f}%")

    fig2, ax2 = plt.subplots(figsize=(3,3))
    ax2.pie(weights, labels=["LSTM","EGARCH","LGB"], autopct="%1.0f%%", startangle=90)
    ax2.set_title("Weight Split")
    st.sidebar.pyplot(fig2)

    # ===== 9. 指标表格 =====
    metrics = {
        "LSTM": {
            "RMSE": np.sqrt(mean_squared_error(p["eval_actual_lstm"], p["lstm_pred"])),
            "MAE": mean_absolute_error(p["eval_actual_lstm"], p["lstm_pred"]),
            "MAPE": mape(p["eval_actual_lstm"], p["lstm_pred"]),
            "R2": r2_score(p["eval_actual_lstm"], p["lstm_pred"]),
            "Direction Acc": direction_accuracy(p["eval_actual_lstm"], p["lstm_pred"]),
        },
        "EGARCH": {
            "RMSE": np.sqrt(mean_squared_error(p["eval_actual_egarch"], p["eg_pred"])),
            "MAE": mean_absolute_error(p["eval_actual_egarch"], p["eg_pred"]),
            "MAPE": mape(p["eval_actual_egarch"], p["eg_pred"]),
            "R2": r2_score(p["eval_actual_egarch"], p["eg_pred"]),
            "Direction Acc": direction_accuracy(p["eval_actual_egarch"], p["eg_pred"]),
        },
        "LightGBM": {
            "RMSE": np.sqrt(mean_squared_error(p["eval_actual_lgb"], p["lgb_pred"])),
            "MAE": mean_absolute_error(p["eval_actual_lgb"], p["lgb_pred"]),
            "MAPE": mape(p["eval_actual_lgb"], p["lgb_pred"]),
            "R2": r2_score(p["eval_actual_lgb"], p["lgb_pred"]),
            "Direction Acc": direction_accuracy(p["eval_actual_lgb"], p["lgb_pred"]),
        },
        "Ensemble(user)": {
            "RMSE": np.sqrt(mean_squared_error(p["eval_actual_lgb"], blend_pred)),
            "MAE": mean_absolute_error(p["eval_actual_lgb"], blend_pred),
            "MAPE": mape(p["eval_actual_lgb"], blend_pred),
            "R2": r2_score(p["eval_actual_lgb"], blend_pred),
            "Direction Acc": direction_accuracy(p["eval_actual_lgb"], blend_pred),
        },
        "Ensemble(opt)": {
            "RMSE": np.sqrt(mean_squared_error(p["eval_actual_lgb"], ensemble_opt)),
            "MAE": mean_absolute_error(p["eval_actual_lgb"], ensemble_opt),
            "MAPE": mape(p["eval_actual_lgb"], ensemble_opt),
            "R2": r2_score(p["eval_actual_lgb"], ensemble_opt),
            "Direction Acc": direction_accuracy(p["eval_actual_lgb"], ensemble_opt),
        },
    }

    # ===== 10. 绘图 =====
    lines_to_show = st.multiselect(
        "Select lines to show",
        ["Actual", "LSTM", "EGARCH", "LightGBM", "Ensemble(user)", "Ensemble(opt)"],
        default=["Actual", "LSTM", "EGARCH", "LightGBM", "Ensemble(user)", "Ensemble(opt)"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 确保所有数据长度一致
    min_len = min(len(p["dates"]), len(p["eval_actual_lgb"]))
    dates = p["dates"].iloc[-min_len:]
    actual = p["eval_actual_lgb"][-min_len:]
    
    if "Actual" in lines_to_show:
        ax.plot(dates, actual, label="Actual", linewidth=2)
    
    if "LSTM" in lines_to_show:
        lstm_data = np.concatenate([p["hist_lstm"], p["lstm_pred"]])
        lstm_dates = p["dates"].iloc[-len(lstm_data):]
        ax.plot(lstm_dates, lstm_data, linestyle="--", label="LSTM")
    
    if "EGARCH" in lines_to_show:
        egarch_data = np.concatenate([p["hist_egarch"], p["eg_pred"]])
        egarch_dates = p["dates"].iloc[-len(egarch_data):]
        ax.plot(egarch_dates, egarch_data, linestyle=":", label="EGARCH")
    
    if "LightGBM" in lines_to_show:
        lgb_data = np.concatenate([p["hist_lgb"], p["lgb_pred"]])
        lgb_dates = p["dates"].iloc[-len(lgb_data):]
        ax.plot(lgb_dates, lgb_data, linestyle="-.", label="LightGBM", color="purple")
    
    if mode == "Manual" and "Ensemble(user)" in lines_to_show:
        ensemble_data = np.concatenate([p["hist_lgb"], blend_pred])
        ensemble_dates = p["dates"].iloc[-len(ensemble_data):]
        ax.plot(ensemble_dates, ensemble_data, linestyle="-", label="Ensemble(user)", color="black", linewidth=2)
    
    if "Ensemble(opt)" in lines_to_show:
        ensemble_opt_data = np.concatenate([p["hist_lgb"], ensemble_opt])
        ensemble_opt_dates = p["dates"].iloc[-len(ensemble_opt_data):]
        ax.plot(ensemble_opt_dates, ensemble_opt_data, linestyle="-", label="Ensemble(opt)", color="red", linewidth=2)
    
    # 添加垂直线标记预测开始点
    forecast_start_idx = len(p["hist_lgb"])
    ax.axvline(p["dates"].iloc[forecast_start_idx], color='gray', linestyle=':', linewidth=1)
    
    ax.set_title("Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    st.pyplot(fig)

    # ===== 11. 显示指标 =====
    st.subheader("Forecast Accuracy Metrics")
    for model_name, stats in metrics.items():
        st.write(f"**{model_name}**: RMSE = {stats['RMSE']:.2f}, MAE = {stats['MAE']:.2f}, MAPE = {stats['MAPE']:.2f}%, R2 = {stats['R2']:.2f}, Direction Acc = {stats['Direction Acc']:.2f}")

else:
    st.info("Please upload a Tesla stock CSV file to begin.")
