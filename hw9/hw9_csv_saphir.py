import pandas as pd 
import numpy as np

df = pd.read_csv('hw9_data.csv',usecols=['Δt (hr)', 'p̄ws (psia)'])
tp = 2160
pwf = df['p̄ws (psia)'].iloc[0]
df['Δt (hr)'] = df['Δt (hr)'] + tp
producing_time = np.arange(0, 2160, 1)
pws_producing = np.zeros_like(producing_time)
pws_producing = pwf 
df_producing = pd.DataFrame({'Δt (hr)': producing_time, 'p̄ws (psia)': pws_producing})
df_combined = pd.concat([df_producing, df], ignore_index=True)
df_combined.to_csv('hw9_data_saphir.csv', index=False)