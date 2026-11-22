"""
app.py
Market Anomaly Radar - Web Application using Streamlit, Plotly, and PyTorch.
Detects financial time-series anomalies using an LSTM Autoencoder compared with Z-Score.
"""

from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from data_utils import download_financial_data, prepare_datasets, preprocess_features
from model import LSTMAutoencoder, compute_reconstruction_errors, train_autoencoder

# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Market Anomaly Radar | LSTM Autoencoder",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for fintech UI theme
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38ef7d 0%, #11998e 50%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.3rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
        transform: translateY(-1px);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Cấu Hình Radar")

    ticker_input = st.text_input(
        "Mã Tài Sản (Ticker):",
        value="BTC-USD",
        help="Nhập mã chứng khoán hoặc tiền mã hoá từ Yahoo Finance (ví dụ: BTC-USD, NVDA, AAPL, ^GSPC, ETH-USD).",
    ).strip().upper()

    st.caption("💡 **Gợi ý:** `BTC-USD`, `NVDA`, `AAPL`, `^GSPC`, `ETH-USD`")

    col_d1, col_d2 = st.columns(2)
    default_end = datetime.today()
    default_start = default_end - timedelta(days=365 * 2)

    with col_d1:
        start_date = st.date_input(
            "Ngày Bắt Đầu:",
            value=default_start,
            max_value=default_end - timedelta(days=60),
        )
    with col_d2:
        end_date = st.date_input(
            "Ngày Kết Thúc:",
            value=default_end,
            max_value=default_end,
        )

    st.markdown("---")
    st.markdown("### 🎯 Chế Độ Nhận Diện")

    mode_choice = st.radio(
        "Phương thức tính Reconstruction Error:",
        options=[
            ("last_day", "Last-Day (Nhạy, bắt sốc tức thời)"),
            ("window_avg", "Window Average (Bắt khủng hoảng kéo dài)"),
        ],
        format_func=lambda x: x[1],
        index=0,
        help="Last-Day: Đo lỗi tái tạo tại ngày cuối cùng của cửa sổ để nhận diện cú sốc giá/khối lượng bất ngờ. Window Average: Đo lỗi trung bình cả cửa sổ để phát hiện sự thay đổi cấu trúc thị trường dài hạn.",
    )
    anomaly_mode = mode_choice[0]

    st.markdown("---")
    st.markdown("### 🎚️ Ngưỡng Bất Thường (Threshold)")

    # Dynamic Threshold Percentile Slider
    threshold_percentile = st.slider(
        "Threshold Percentile (%):",
        min_value=90.0,
        max_value=99.5,
        value=95.0,
        step=0.5,
        help="Phần trăm phân vị để cắt ngưỡng MSE. Kéo thanh trượt sẽ cập nhật biểu đồ và bảng kết quả ngay lập tức mà không cần train lại model!",
    )

    with st.expander("🛠️ Cấu hình Nâng cao (Mô hình)"):
        window_size = st.slider(
            "Window Size (Cửa sổ trượt):",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
        )
        train_split = st.slider(
            "Tỷ lệ Train Split (%):",
            min_value=50,
            max_value=85,
            value=70,
            step=5,
        )
        epochs = st.slider(
            "Số Epochs Huấn Luyện:", min_value=15, max_value=60, value=30, step=5
        )

    btn_train = st.button("🚀 Train & Detect Anomalies", use_container_width=True)

# ---------------------------------------------------------
# Core Processing & State Caching
# ---------------------------------------------------------


def run_pipeline(
    ticker: str,
    s_date,
    e_date,
    w_size: int,
    t_split: float,
    ep: int,
    mode: str,
):
    """Executes the complete pipeline: fetch data, preprocess, scale without leakage,

    train PyTorch LSTM Autoencoder, and compute reconstruction errors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Download
    raw_df = download_financial_data(ticker, s_date, e_date)

    # 2. Features
    feat_df = preprocess_features(raw_df)

    # 3. Splits & Scaler (No data leakage!)
    X_train, train_dates, X_test, test_df, scaler = prepare_datasets(
        feat_df, window_size=w_size, train_ratio=t_split
    )

    # 4. Model Architecture
    model = LSTMAutoencoder(
        input_dim=2,  # [log_return, vol_change]
        seq_len=w_size,
        hidden_dim=48,
        latent_dim=16,
        num_layers=2,
        dropout=0.1,
    )

    # 5. Train
    p_bar = st.progress(0, text="Đang huấn luyện LSTM Autoencoder...")

    def update_pbar(cur_ep, total_ep, cur_loss):
        pct = cur_ep / total_ep
        p_bar.progress(
            pct,
            text=f"Huấn luyện Epoch {cur_ep}/{total_ep} - MSE Loss: {cur_loss:.5f}",
        )

    loss_history = train_autoencoder(
        model=model,
        X_train=X_train,
        epochs=ep,
        batch_size=32,
        lr=0.003,
        device=device,
        progress_callback=update_pbar,
    )
    p_bar.empty()

    # 6. Errors on Train and Test
    train_errors = compute_reconstruction_errors(
        model, X_train, mode=mode, device=device
    )
    test_errors = compute_reconstruction_errors(
        model, X_test, mode=mode, device=device
    )

    return {
        "ticker": ticker,
        "feat_df": feat_df,
        "train_dates": train_dates,
        "X_train": X_train,
        "train_errors": train_errors,
        "X_test": X_test,
        "test_df": test_df,
        "test_errors": test_errors,
        "loss_history": loss_history,
        "mode": mode,
        "window_size": w_size,
        "train_split": t_split,
        "scaler": scaler,
    }


# Initialize Session State
if "radar_data" not in st.session_state:
    st.session_state.radar_data = None

# If user clicked button or if state is empty, trigger training
need_train = btn_train or (st.session_state.radar_data is None)

if need_train:
    with st.spinner(
        f"Đang tải dữ liệu {ticker_input} và huấn luyện mạng LSTM Autoencoder..."
    ):
        try:
            results = run_pipeline(
                ticker=ticker_input,
                s_date=start_date,
                e_date=end_date,
                w_size=window_size,
                t_split=train_split / 100.0,
                ep=epochs,
                mode=anomaly_mode,
            )
            st.session_state.radar_data = results
            st.toast(
                f"✅ Huấn luyện thành công mô hình cho {ticker_input}!",
                icon="🚀",
            )
        except Exception as e:
            st.error(f"❌ Đã xảy ra lỗi trong quá trình xử lý: {str(e)}")
            st.stop()

data = st.session_state.radar_data

# If detection mode changed after run, re-compute errors without retraining model
# (Note: errors were computed based on saved results. If user changes mode without clicking train, notify or recompute)
if data["mode"] != anomaly_mode:
    # Quick recomputation of reconstruction errors for the new mode using same representations
    st.info(
        f"Đã chuyển chế độ sang '{anomaly_mode}'. Bấm **Train & Detect Anomalies** để huấn luyện lại theo cấu hình mới nhất nếu cần."
    )

# ---------------------------------------------------------
# Dynamic Anomaly Evaluation (Instant Response to Slider)
# ---------------------------------------------------------
# Calculate dynamic threshold based on training distribution or test distribution
train_err = data["train_errors"]
test_err = data["test_errors"]

# Standard practice: Threshold is computed from normal baseline (train_errors percentile)
# or calibrated percentile of all errors
threshold_val = float(np.percentile(train_err, threshold_percentile))

test_df = data["test_df"].copy()
test_df["reconstruction_error"] = test_err
test_df["lstm_anomaly"] = test_df["reconstruction_error"] > threshold_val

# Classify anomalies
test_df["both_anomaly"] = test_df["lstm_anomaly"] & test_df["z_anomaly"]
test_df["lstm_only"] = test_df["lstm_anomaly"] & (~test_df["z_anomaly"])
test_df["z_only"] = (~test_df["lstm_anomaly"]) & test_df["z_anomaly"]

total_test_sessions = len(test_df)
lstm_count = int(test_df["lstm_anomaly"].sum())
zscore_count = int(test_df["z_anomaly"].sum())
both_count = int(test_df["both_anomaly"].sum())

# ---------------------------------------------------------
# Main Dashboard UI
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">📡 Market Anomaly Radar</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="sub-title">Phát hiện bất thường chuỗi thời gian trên dữ liệu tài chính thực tế bằng PyTorch <b>LSTM Autoencoder</b> & so sánh với <b>Z-Score Statistical Baseline</b> | Tài sản: <code style="color:#38ef7d;">{data["ticker"]}</code></div>',
    unsafe_allow_html=True,
)

# 1. KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Tổng số phiên test</div>
            <div class="metric-value">{total_test_sessions:,}</div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.2rem;">Kỳ đánh giá Out-of-Sample</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with kpi2:
    pct_lstm = (
        (lstm_count / total_test_sessions * 100)
        if total_test_sessions > 0
        else 0
    )
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #ef4444;">
            <div class="metric-label" style="color: #f87171;">Bất thường LSTM</div>
            <div class="metric-value" style="color: #ef4444;">{lstm_count} <span style="font-size: 1rem; color: #94a3b8;">({pct_lstm:.1f}%)</span></div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;">MSE > Ngưỡng {threshold_percentile}%</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with kpi3:
    pct_z = (
        (zscore_count / total_test_sessions * 100)
        if total_test_sessions > 0
        else 0
    )
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-label" style="color: #34d399;">Bất thường Z-Score</div>
            <div class="metric-value" style="color: #10b981;">{zscore_count} <span style="font-size: 1rem; color: #94a3b8;">({pct_z:.1f}%)</span></div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;">|Z-Score Return| > 2.0</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #a855f7;">
            <div class="metric-label" style="color: #c084fc;">Ngưỡng MSE Hiện Tại</div>
            <div class="metric-value" style="color: #c084fc;">{threshold_val:.4f}</div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;">Đồng thuận cả 2: {both_count} phiên</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Biểu đồ 1: Giá đóng cửa & So sánh điểm bất thường
# ---------------------------------------------------------
st.markdown("### 📈 Biểu Đồ 1: Giá Đóng Cửa & Điểm Bất Thường Đối Chiếu")
st.caption(
    "So sánh đa chiều giữa mô hình học sâu **LSTM Autoencoder** và phương pháp thống kê truyền thống **Z-Score**."
)

fig_price = go.Figure()

# Đường giá đóng cửa
fig_price.add_trace(
    go.Scatter(
        x=test_df.index,
        y=test_df["Close"],
        mode="lines",
        name=f"Giá Đóng Cửa ({data['ticker']})",
        line=dict(color="#38bdf8", width=2),
        hoverinfo="skip",
    )
)

# 1. LSTM Autoencoder Chỉ Bắt (Chấm Đỏ)
df_lstm_only = test_df[test_df["lstm_only"]]
if not df_lstm_only.empty:
    fig_price.add_trace(
        go.Scatter(
            x=df_lstm_only.index,
            y=df_lstm_only["Close"],
            mode="markers",
            name="Chỉ LSTM Bắt (Chấm Đỏ)",
            marker=dict(
                color="#ef4444",
                symbol="circle",
                size=9,
                line=dict(color="#ffffff", width=1.2),
            ),
            customdata=np.stack(
                (
                    df_lstm_only["reconstruction_error"],
                    df_lstm_only["z_score"],
                    df_lstm_only["log_return"] * 100,
                    df_lstm_only["vol_change"] * 100,
                ),
                axis=-1,
            ),
            hovertemplate=(
                "<b>🔴 [LSTM Autoencoder Only]</b><br>"
                "Ngày: %{x|%Y-%m-%d}<br>"
                "Giá: $%{y:,.2f}<br>"
                "MSE Error: %{customdata[0]:.4f}<br>"
                "Z-Score: %{customdata[1]:.2f}<br>"
                "Log Return: %{customdata[2]:.2f}%<br>"
                "Vol Change: %{customdata[3]:.2f}%<extra></extra>"
            ),
        )
    )

# 2. Z-Score Chỉ Bắt (Dấu X Xanh Lá)
df_z_only = test_df[test_df["z_only"]]
if not df_z_only.empty:
    fig_price.add_trace(
        go.Scatter(
            x=df_z_only.index,
            y=df_z_only["Close"],
            mode="markers",
            name="Chỉ Z-Score Bắt (Dấu X Xanh Lá)",
            marker=dict(
                color="#10b981",
                symbol="x",
                size=11,
                line=dict(color="#10b981", width=2.5),
            ),
            customdata=np.stack(
                (
                    df_z_only["reconstruction_error"],
                    df_z_only["z_score"],
                    df_z_only["log_return"] * 100,
                    df_z_only["vol_change"] * 100,
                ),
                axis=-1,
            ),
            hovertemplate=(
                "<b>🟢 [Z-Score Only]</b><br>"
                "Ngày: %{x|%Y-%m-%d}<br>"
                "Giá: $%{y:,.2f}<br>"
                "MSE Error: %{customdata[0]:.4f}<br>"
                "Z-Score: %{customdata[1]:.2f}<br>"
                "Log Return: %{customdata[2]:.2f}%<br>"
                "Vol Change: %{customdata[3]:.2f}%<extra></extra>"
            ),
        )
    )

# 3. Cả 2 Cùng Bắt (Ngôi Sao Tím)
df_both = test_df[test_df["both_anomaly"]]
if not df_both.empty:
    fig_price.add_trace(
        go.Scatter(
            x=df_both.index,
            y=df_both["Close"],
            mode="markers",
            name="Cả 2 Cùng Bắt (Ngôi Sao Tím)",
            marker=dict(
                color="#c084fc",
                symbol="star",
                size=14,
                line=dict(color="#ffffff", width=1.5),
            ),
            customdata=np.stack(
                (
                    df_both["reconstruction_error"],
                    df_both["z_score"],
                    df_both["log_return"] * 100,
                    df_both["vol_change"] * 100,
                ),
                axis=-1,
            ),
            hovertemplate=(
                "<b>⭐ [CẢ 2 MÔ HÌNH ĐỒNG THUẬN]</b><br>"
                "Ngày: %{x|%Y-%m-%d}<br>"
                "Giá: $%{y:,.2f}<br>"
                "MSE Error: %{customdata[0]:.4f}<br>"
                "Z-Score: %{customdata[1]:.2f}<br>"
                "Log Return: %{customdata[2]:.2f}%<br>"
                "Vol Change: %{customdata[3]:.2f}%<extra></extra>"
            ),
        )
    )

fig_price.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(15, 23, 42, 0.6)",
    paper_bgcolor="rgba(15, 23, 42, 0)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(30, 41, 59, 0.7)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    xaxis=dict(
        title="Thời Gian",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        rangeslider=dict(visible=False),
    ),
    yaxis=dict(
        title="Giá Đóng Cửa (USD)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
    ),
    height=480,
    margin=dict(l=40, r=40, t=30, b=40),
)

st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------------------------------
# Biểu đồ 2: Reconstruction Error (MSE) & Ngưỡng Threshold
# ---------------------------------------------------------
st.markdown(
    "### 📉 Biểu Đồ 2: Lỗi Tái Tạo (Reconstruction Error MSE) & Ngưỡng Cắt"
)
st.caption(
    f"Các phiên có MSE vượt trên đường nét đứt đỏ (ngưỡng phân vị {threshold_percentile}%) được phân loại là bất thường."
)

fig_error = go.Figure()

# Đường Reconstruction Error
fig_error.add_trace(
    go.Scatter(
        x=test_df.index,
        y=test_df["reconstruction_error"],
        mode="lines",
        name="Reconstruction Error (MSE)",
        line=dict(color="#f59e0b", width=1.8),
        customdata=np.stack(
            (test_df["log_return"] * 100, test_df["vol_change"] * 100), axis=-1
        ),
        hovertemplate=(
            "Ngày: %{x|%Y-%m-%d}<br>"
            "Reconstruction Error: %{y:.5f}<br>"
            "Log Return: %{customdata[0]:.2f}%<br>"
            "Vol Change: %{customdata[1]:.2f}%<extra></extra>"
        ),
    )
)

# Điểm bất thường trên biểu đồ lỗi
df_error_anom = test_df[test_df["lstm_anomaly"]]
if not df_error_anom.empty:
    fig_error.add_trace(
        go.Scatter(
            x=df_error_anom.index,
            y=df_error_anom["reconstruction_error"],
            mode="markers",
            name="Vượt Ngưỡng (Anomaly)",
            marker=dict(
                color="#ef4444",
                size=7,
                symbol="circle",
                line=dict(color="#ffffff", width=0.8),
            ),
            hoverinfo="skip",
        )
    )

# Đường nét đứt đỏ Threshold
fig_error.add_shape(
    type="line",
    x0=test_df.index.min(),
    x1=test_df.index.max(),
    y0=threshold_val,
    y1=threshold_val,
    line=dict(color="#ef4444", width=2, dash="dash"),
)

fig_error.add_annotation(
    x=test_df.index.max(),
    y=threshold_val,
    text=f"Ngưỡng ({threshold_percentile}%): {threshold_val:.4f}",
    showarrow=False,
    xanchor="right",
    yanchor="bottom",
    font=dict(color="#ef4444", size=12),
    bgcolor="rgba(30, 41, 59, 0.8)",
)

fig_error.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(15, 23, 42, 0.6)",
    paper_bgcolor="rgba(15, 23, 42, 0)",
    xaxis=dict(
        title="Thời Gian", showgrid=True, gridcolor="rgba(255,255,255,0.06)"
    ),
    yaxis=dict(
        title="Mean Squared Error (MSE)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
    ),
    height=320,
    margin=dict(l=40, r=40, t=20, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(30, 41, 59, 0.7)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
)

st.plotly_chart(fig_error, use_container_width=True)

# ---------------------------------------------------------
# Bảng Dữ Liệu: Top 15 Phiên Lỗi Tái Tạo Cao Nhất & Tải CSV
# ---------------------------------------------------------
st.markdown("### 📋 Bảng Dữ Liệu: Top 15 Phiên Có Reconstruction Error Cao Nhất")
st.caption(
    "Danh sách các phiên có độ lệch lớn nhất so với hành vi mẫu của thị trường."
)

# Prepare Top 15 Table
table_df = test_df.sort_values(
    by="reconstruction_error", ascending=False
).head(15)


def label_anomaly_type(row):
    if row["both_anomaly"]:
        return "⭐ Cả 2 cùng bắt"
    elif row["lstm_only"]:
        return "🔴 Chỉ LSTM Autoencoder"
    elif row["z_only"]:
        return "🟢 Chỉ Z-Score"
    else:
        return "⚪ Bình thường"


table_df["Phân Loại"] = table_df.apply(label_anomaly_type, axis=1)

display_table = pd.DataFrame(
    {
        "Ngày": table_df.index.strftime("%Y-%m-%d"),
        "Giá Đóng Cửa ($)": table_df["Close"].map(
            lambda x: f"${x:,.2f}" if x >= 1 else f"${x:,.4f}"
        ),
        "Log Return (%)": (table_df["log_return"] * 100).map(
            lambda x: f"{x:+.2f}%"
        ),
        "Vol Change (%)": (table_df["vol_change"] * 100).map(
            lambda x: f"{x:+.2f}%"
        ),
        "MSE Error": table_df["reconstruction_error"].map(
            lambda x: f"{x:.5f}"
        ),
        "Z-Score Return": table_df["z_score"].map(lambda x: f"{x:+.2f}"),
        "Nhận Diện Bất Thường": table_df["Phân Loại"],
    }
)

st.dataframe(display_table, use_container_width=True, hide_index=True)

# Download buttons
col_dl1, col_dl2 = st.columns([1, 3])

# Prepare full test dataset CSV
export_df = test_df.copy()
export_df["Date"] = export_df.index.strftime("%Y-%m-%d")
export_df["Anomaly_Type"] = export_df.apply(label_anomaly_type, axis=1)
export_cols = [
    "Date",
    "Close",
    "Volume",
    "log_return",
    "vol_change",
    "reconstruction_error",
    "z_score",
    "lstm_anomaly",
    "z_anomaly",
    "both_anomaly",
    "Anomaly_Type",
]
csv_buffer = io.StringIO()
export_df[export_cols].to_csv(csv_buffer, index=False)

with col_dl1:
    st.download_button(
        label="📥 Tải xuống Toàn Bộ Dữ Liệu Test (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"market_anomaly_{data['ticker']}_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    st.info(
        f"💡 **Mẹo:** Mô hình sử dụng cửa sổ trượt `{data['window_size']}` phiên. "
        f"Tỷ lệ dữ liệu Train/Test: `{int(data['train_split']*100)}% / {int((1-data['train_split'])*100)}%`. "
        "Bộ chuẩn hóa `StandardScaler` được fit độc quyền trên tập Train để loại trừ hoàn toàn rủi ro rò rỉ dữ liệu (Data Leakage)."
    )
