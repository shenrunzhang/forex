# forex

## File descriptions

applyindicators.py - Applies 8 different technical indicators to raw data, and outputs data.csv. Note that the first few lines of the output file will have some NaN values some indicators require a few time steps before starting.

lstm.py - Takes what applyindicators.py outputted and trains a lstm model, then saves the model into mymodel file. 

plot.py - Runs predictions from a saved model. 
