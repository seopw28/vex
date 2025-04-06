
#%%
import dash
from dash import html, dcc
import yfinance as yf
import plotly.graph_objs as go

app = dash.Dash(__name__)
df = yf.download('AAPL', period='1y')

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='AAPL'))

app.layout = html.Div([
    html.H1("애플 주가 대시보드"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)

# %%
