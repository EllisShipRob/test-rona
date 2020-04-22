
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv', 
  parse_dates=[0],
  index_col=False,
)

df[df.state == 'Washington'].plot(x='date', y='cases')
plt.show()

print(df.head())
