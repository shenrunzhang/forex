# forex

## File descriptions

applyindicators.py - Applies 8 different technical indicators to raw data, and outputs data.csv. Note that the first few lines of the output file will have some NaN values some indicators require a few time steps before starting.

lstm.py - Takes what applyindicators.py outputted and trains a lstm model, then saves the model into mymodel file. 

plot.py - Runs predictions from a saved model. 

Threshold.py - has function get_threshold() to get best threshold when given pandas dataframe of closing prices



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