import numpy as np
import pandas as pd

a1 = np.array([11,22,33])
a2 = np.array([44,55,66])

a1 = pd.DataFrame([a1,a2]).transpose()
a1.columns = ["a1","a2"]
print(a1)