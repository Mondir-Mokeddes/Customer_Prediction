import pandas as pd
script_location = Path(__file__).absolute().parent.parent
df = pd.read_csv(script_location / 'Data/Processed/Order_and_departments.csv')

df = pd.crosstab(df['order_id'], df['department_id'])


df.to_csv(script_location  /'Data/Processed/Order_and_departments_count.csv')
