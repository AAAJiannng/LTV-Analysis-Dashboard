import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="用户LTV预测仪表板", layout="wide")
st.title("📊用户生命周期价值（LTV）预测")

# 直接读取本地文件
df_sales = pd.read_csv("电商历史订单.csv", encoding="utf8")
st.success(f"✅已加载数据，共{len(df_sales)}条记录")

# 预览数据
st.subheader("📋原始数据预览")
st.dataframe(df_sales.head())

# 数据预处理（与之前完全一致）
with st.spinner("正在处理数据..."):
    df_sales['总价'] = df_sales['数量'] * df_sales['单价']
    df_sales['消费日期'] = pd.to_datetime(df_sales['消费日期'])
    
    min_date = df_sales['消费日期'].min()
    max_date = df_sales['消费日期'].max()
    st.info(f"数据日期范围：{min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}")
    
    three_months_later = min_date + pd.DateOffset(months=3)
    df_sales_3m = df_sales[(df_sales['消费日期'] >= min_date) & (df_sales['消费日期'] < three_months_later)]
    if df_sales_3m.empty:
        st.error("数据不足3个月，无法计算RFM特征。")
        st.stop()
    st.success(f"前3个月数据量：{len(df_sales_3m)} 条记录")

    # RFM计算
    df_R = df_sales_3m.groupby('用户码')['消费日期'].max().reset_index()
    df_R.columns = ['用户码', '最近购买日期']
    latest_date = df_R['最近购买日期'].max()
    df_R['R值'] = (latest_date - df_R['最近购买日期']).dt.days

    df_F = df_sales_3m.groupby('用户码')['消费日期'].count().reset_index()
    df_F.columns = ['用户码', 'F值']

    df_M = df_sales_3m.groupby('用户码')['总价'].sum().reset_index()
    df_M.columns = ['用户码', 'M值']

    df_RFM = df_R[['用户码', 'R值']].merge(df_F, on='用户码').merge(df_M, on='用户码')

    # 年度LTV
    df_annual = df_sales.groupby('用户码')['总价'].sum().reset_index()
    df_annual.columns = ['用户码', '年度LTV']

    df_LTV = df_RFM.merge(df_annual, on='用户码', how='left').dropna(subset=['年度LTV'])
    st.success(f"有效用户数：{len(df_LTV)}")

    X = df_LTV[['R值', 'F值', 'M值']]
    y = df_LTV['年度LTV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

# 模型性能
st.subheader("📈模型性能")
col1, col2, col3 = st.columns(3)
col1.metric("训练集R²", f"{r2_train:.4f}")
col2.metric("测试集R²", f"{r2_test:.4f}")
col3.metric("模型截距", f"{model.intercept_:.2f}")

# 系数
coef_df = pd.DataFrame({
    '特征': ['R值', 'F值', 'M值'],
    '系数': model.coef_
})
st.dataframe(coef_df)

# 散点图
st.subheader("📉预测值与实际值对比（测试集）")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test_pred, alpha=0.6)
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('实际值')
ax.set_ylabel('预测值')
ax.set_title('测试集：实际值vs预测值')
st.pyplot(fig)

# 新用户预测
st.subheader("🔮新用户LTV预测")
col1, col2, col3 = st.columns(3)
with col1:
    r = st.number_input("R值（最近购买天数）", min_value=0, value=30)
with col2:
    f = st.number_input("F值（购买次数）", min_value=1, value=5)
with col3:
    m = st.number_input("M值（前3个月消费总额）", min_value=0.0, value=500.0)

if st.button("预测年度LTV"):
    input_df = pd.DataFrame([[r, f, m]], columns=['R值', 'F值', 'M值'])
    pred = model.predict(input_df)[0]
    st.success(f"预测的年度LTV为：**¥{pred:.2f}**")