#%% Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import warnings

# Suppress Streamlit warnings
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

#%% Title
st.title("애플 주가")

#%% Download data
@st.cache_data
def load_data():
    return yf.download('AAPL', period='6mo')

#%% Show data
df = load_data()
# Fix column names if they have multi-index
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
st.line_chart(df['Close'])

#%% Additional visualization
st.subheader("상세 데이터")
st.dataframe(df)

#%% Interactive chart
fig = px.line(df, y='Close', title='애플 주가 추이')
st.plotly_chart(fig)

#%% Sidebar controls
st.sidebar.header("설정")
period = st.sidebar.selectbox(
    "기간 선택",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
)

if period != "6mo":
    df = yf.download('AAPL', period=period)
    # Fix column names if they have multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    st.line_chart(df['Close'])
# %%
