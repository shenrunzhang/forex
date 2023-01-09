# forex

## File descriptions

applyindicators.py - Applies 8 different technical indicators to raw data, and outputs data.csv. Note that the first few lines of the output file will have some NaN values some indicators require a few time steps before starting.

lstm.py - Takes what applyindicators.py outputted and trains a lstm model, then saves the model into mymodel file. 

plot.py - Runs predictions from a saved model. 

Threshold.py - has function get_threshold() to get best threshold when given pandas dataframe of closing prices

## Apply Indicators

In day trading, technical indiactors are mathematically derived patterns based on historical price data that are used to determine whether a stock or currency pair is over or under bought. They are used alongside financial indicators to tell whether to buy or sell at any point. 

In our technical model, 7 different technical indicators are used along with the closing price as the training dataset. 

* MA with a period of 10
* MACD with short- and long-term periods of 12 and 26, respectively
* ROC with a period of 2
* Momentum with a period of 4
* RSI with a period of 10
* BB with period of 20
* CCI with a period of 20

To derive these indicators, we start with the Opening, High, Low and Closing price data.





## Debug

I always get an import error when following tensorflow tutorials. To fix it I changed 
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
```
to 
```
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
```
So if you are encountering something similar you can try to change it back. 
