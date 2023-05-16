# Import your libraries
import pandas as pd

# Start writing code
airbnb_contacts.head()

df = airbnb_contacts

df = df.groupby('id_guest', as_index = False)['n_messages'].sum().sort_values(by = 'n_messages', ascending = False)

# First approach: Transform method 
df['rank'] = df[['n_messages']].transform('rank', method = 'dense', ascending = False)

# Second approach: Assign method 
df.assign(rank=df['n_messages'].rank(method='dense',ascending=False))

df = df.reset_index(drop = True)
df.iloc[:, [2, 0, 1]]
