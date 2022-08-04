import pickle

import pandas as pd

pkl_file = "..\Dataset\Packaged Pkl\input_data_with_matrix_1.5_control_data_type1_2724.pkl"
f = open(pkl_file, 'rb')
data = pickle.load(f)
df = pd.DataFrame(data)
print(df)
# df.to_csv("data_input.csv")
