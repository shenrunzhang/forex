import pandas as pd
# MA with a period of 10
# MACD with short- and long-term periods of 12 and 26, respectively
# ROC with a period of 2
# Momentum with a period of 4
# RSI with a period of 10
# BB with period of 20
# CCI with a period of 20

df = pd.read_csv(
    r'E:\PC_nou\trading_ai\forex_shenny\forex\fundamental_data\EURUSD_Candlestick_1_D_BID_01.01.2012-10.12.2022.csv')

data = pd.DataFrame

data = df[['Close']].copy()

# Add moving average, window = 10
data['MA10'] = df['Close'].rolling(10).mean()

# Add MACD
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
data['MACD'] = macd

# Add ROC, period = 2
n_steps = 2


def my_fun(x):
    return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]


data['ROC'] = data['Close'].rolling(n_steps).apply(my_fun)

# Add Momentum, period = 4
n_steps = 4


def my_fun(x):
    return (x.iloc[-1] - x.iloc[0])


data['Momentum'] = data['Close'].rolling(n_steps).apply(my_fun)

# Add RSI
# get the price diff
delta = data['Close'].diff()

# positive gains (up) and negative gains (down) Series
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0

period = 10

_gain = up.ewm(alpha=1.0 / period, adjust=True).mean()
_loss = down.abs().ewm(alpha=1.0 / period, adjust=True).mean()
RS = _gain / _loss

data["RSI"] = 100 - (100 / (1 + RS))

# Add Bollinger Bands , period = 20

typical_price = (df['Close'] + df['Low'] + df['High'])/3
std = typical_price.rolling(20).std(ddof=0)
ma_tp = typical_price.rolling(20).mean()
data['BOLU'] = ma_tp + 2*std
data['BOLD'] = ma_tp - 2*std

# Add CCI Commodity channel index, period = 20
tp_rolling = typical_price.rolling(20)

# calculate mean deviation
mad = tp_rolling.apply(lambda s: abs(s - s.mean()).mean(), raw=True)

data["CCI"] = (typical_price - tp_rolling.mean()) / (0.015 * mad)

data.to_csv('technical_data_eurusd2.csv')
