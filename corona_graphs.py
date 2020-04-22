
import pandas as pd
import matplotlib.pyplot as plt
import pylab as plot
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b):
    return a * np.power(b, x)

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def generalized_logistic(x, K, A, C, Q, B, nu):
    return (K - A) / np.power(C + Q * np.exp(-B * x), 1/nu)

params = {  
    'legend.fontsize': 20,
    'legend.handlelength': 2,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'lines.linewidth': 3,
    'lines.markersize': 12,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'xtick.major.size': 12,
    'xtick.major.width': 3,
    'ytick.major.size': 12,
    'ytick.major.width': 3,
    'ytick.minor.size': 6,
    'ytick.minor.width': 1.5,
}
plot.rcParams.update(params)

n_since = 30

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv', 
  parse_dates=[0],
  index_col=False,
)
df = df.drop(columns=['fips', 'deaths'])
df_orig = df.copy()

_df_us = df_orig.copy()
_df_us = _df_us.set_index(['date', 'state', 'county'])
_df_us = _df_us.sum(level=['date'])
_df_us = _df_us.reset_index()
_df_us = _df_us[_df_us.cases > n_since]
_df_us['days_since'] = pd.to_datetime(_df_us.date).sub(_df_us.date.min()).dt.days.tolist()
all_x = np.arange(0, _df_us.days_since.max() + 2, step=1)

plot_states = True
plot_counties = False
plot_usa = False

populations = {
    'Ohio': 11.69,
    'Pennsylvania': 12.81,
    'New York': 8.623,
    'Washington': 7.536,
    'California': 39.56,
    'New Jersey': 8.909,
    'Florida': 21.3,
    'Michigan': 9.996,
    'Louisiana': 4.66,
    'Florida': 21.3,
}

print(df_orig.date.tolist()[-1])



fig, ax = plt.subplots(figsize=(18,10))

if plot_usa:
  _df_us.plot(x='days_since', y='cases', ax=ax, label='USA', logy=True, marker='o', color='lightgrey')

# Plot growth lines
x = np.arange(_df_us.days_since.max() + 1)
for r in [0.25, 0.35, 0.50]:
  ax.plot(x, n_since * np.power(2, r * x), linestyle='--', color='lightgrey')

# Plot states
df = df_orig.copy()
df = df.set_index(['date', 'state', 'county'])
df = df.sum(level=['date', 'state'])
df = df.reset_index()
for state, group in df.groupby('state'):  
  if state in [
    'Ohio',
    'Pennsylvania',
    'New York',
    'Washington',
    'California',
    'New Jersey',
    'Pennsylvania',
    'Florida',
    'Michigan',
    'Louisiana',
    'Florida',
  ]:
    _g = group.copy()
    _g['cases_per_million'] = _g.cases / populations[state]
    _g = _g[_g.cases > n_since]
    # _g = _g[_g.cases_per_million > 1]
    _g['days_since'] = pd.to_datetime(_g.date).sub(_g.date.min()).dt.days.tolist()
    _g.plot(x='days_since', y='cases', ax=ax, label=state, logy=True, marker='o')

plt.legend()
plt.xticks(np.arange(0, _df_us.days_since.max() + 2, step=1))
ax.set_xlim(0, 45)
# ax.set_ylim(1, 20000)
ax.set_ylim(10, 250000)
plt.xlabel("Days since %d cases" % n_since)
plt.ylabel("Cases")
plt.grid()
fig.savefig("1.png")




fig, ax = plt.subplots(figsize=(18,10))

# Plot growth lines
x = np.arange(_df_us.days_since.max() + 1)
for r in [0.25, 0.35, 0.50]:
  ax.plot(x, n_since * np.power(2, r * x), linestyle='--', color='lightgrey')

for state, county in [
  ('Pennsylvania', 'Philadelphia'),
  # ('New York', 'New York City'),
  # ('Washington', 'King'),
  ('Ohio', 'Franklin'),
  ('Pennsylvania', 'Allegheny'),
  # ('Tennessee', 'Williamson'),
  # ('Tennessee', 'Davidson'),
  # ('California', 'San Bernardino'),
  # ('California', 'Orange'),
  # ('California', 'Los Angeles'),
  # ('California', 'Ventura'),
  # ('California', 'Riverside'),
  # ('Tennessee', 'Shelby'),
  # ('Tennessee', 'Knox'),
  # ('Michigan', 'Wayne')         
]:
  _df = df_orig.copy()
  _df = _df[(_df.state == state) & (_df.county == county)]
  _df = _df[_df.cases > n_since]
  _df['days_since'] = pd.to_datetime(_df.date).sub(_df.date.min()).dt.days.tolist()

  n = 10
  popt, pcov = curve_fit(func, _df.days_since.tolist()[-n:], _df.cases.tolist()[-n:])
  ax.semilogy(all_x, func(all_x, *popt), linestyle='--')

  _df.plot(x='days_since', y='cases', ax=ax, label=county + (", %.1f%%" % ((popt[1] - 1) * 100)), logy=True, marker='o', color=ax.get_lines()[-1].get_color())

# Graph apperance
plt.legend()
plt.xticks(np.arange(0, 38, step=1))
ax.set_xlim(0, 38)
ax.set_ylim(10, 100000)
plt.xlabel("Days since %d cases" % n_since)
plt.ylabel("Cases")
plt.grid()
fig.savefig("2.png")



fig, ax = plt.subplots(figsize=(18,10))

for state, county in [
  ('Pennsylvania', 'Allegheny'),
  ('Pennsylvania', 'Philadelphia'),
  # ('Tennessee', 'Williamson'),
  # ('Tennessee', 'Davidson'),
  #   ('California', 'San Bernardino'),
  # ('California', 'Orange'),
  # ('California', 'Los Angeles'),
  # ('California', 'Ventura'),
  # ('California', 'Riverside'),

  # ('Tennessee', 'Shelby'),
  # ('Tennessee', 'Knox'),

  # ('New York', 'New York City'),
  ('Washington', 'King'),
  ('Ohio', 'Franklin'),
  # ('Iowa', 'Allamakee')            
]:
  _df = df_orig.copy()
  _df = _df[(_df.state == state) & (_df.county == county)]
  _df = _df[_df.cases > n_since]
  _df['days_since'] = pd.to_datetime(_df.date).sub(_df.date.min()).dt.days.tolist()

  days_since = True
  if days_since:
    _df = _df.drop(columns=['state', 'county', 'date'])
    _df = _df.set_index(['days_since'])
    _df['pct_change'] = _df.pct_change(periods=1) * 100 
    _df = _df.reset_index()
    _df.rolling(7).mean().plot(x='days_since', y='pct_change', ax=ax, label=county, logy=True, marker='o')
  else:
    _df = _df.drop(columns=['state', 'county', 'days_since'])
    _df = _df.set_index(['date'])
    _df['pct_change'] = _df.pct_change(periods=1) * 100 
    _df = _df.rolling(3).mean()
    _df.reset_index()
    _df.plot(y='pct_change', ax=ax, label=county, logy=False, marker='o')

# Graph apperance
plt.legend()
plt.xticks(np.arange(0, 50, step=1))
# plt.yticks(np.arange(0.1, 60, step=1))
ax.set_xlim(0, 50)
ax.set_ylim(0.1, 60)
plt.xlabel("Days since %d cases" % n_since)
plt.ylabel("Growth")
plt.grid()
fig.savefig("3.png")





fig, ax = plt.subplots(figsize=(18,10))

print("starting to graph...")

for state, county in [
  ('Pennsylvania', 'Allegheny'),
  ('Pennsylvania', 'Philadelphia'),
  #('Ohio', 'Franklin'),
  # ('Tennessee', 'Williamson'),

  # ('Tennessee', 'Davidson'),
  # ('Tennessee', 'Shelby'),
  # ('Tennessee', 'Knox'),
  # ('New York', 'New York City')
]:
  _df = df_orig.copy()
  _df = _df[(_df.state == state) & (_df.county == county)]
  _df = _df[_df.cases > 30]
  _df['days_since'] = pd.to_datetime(_df.date).sub(_df.date.min()).dt.days.tolist()

  # popt, pcov = curve_fit(logistic, _df.days_since.tolist(), _df.cases.tolist(), p0=[1200, 0.2, 22])
  # ax.plot(all_x, logistic(all_x, *popt), linestyle='--')
  # popt, pcov = curve_fit(generalized_logistic, _df.days_since.tolist(), _df.cases.tolist(), p0=[261, -246, 0.8, 4.3, 0.15, 0.30], maxfev=20000)
  popt, pcov = curve_fit(generalized_logistic, _df.days_since.tolist(), _df.cases.tolist(), p0=[500, -500, 0.8, 10, 0.15, 0.30], maxfev=20000)
  ax.plot(all_x, generalized_logistic(all_x, *popt), linestyle='--')

  _df.plot(x='days_since', y='cases', ax=ax, label=county, logy=False, marker='o', color=ax.get_lines()[-1].get_color())

# Graph apperance
# plt.legend()
plt.xticks(np.arange(0, 50, step=1))
ax.set_xlim(0, 50)
# ax.set_ylim(0, 2000)
plt.xlabel("Days since %d cases" % n_since)
plt.ylabel("Total Cases")
plt.grid()
fig.savefig("4.png")






fig, ax = plt.subplots(figsize=(18,10))


for state, county in [
  ('Pennsylvania', 'Allegheny'),
  # ('Pennsylvania', 'Philadelphia'),
  # ('Ohio', 'Franklin'),
  # ('Tennessee', 'Williamson'),
  # ('Tennessee', 'Davidson'),
  # ('Tennessee', 'Shelby'),
  # ('Tennessee', 'Knox'),
  # ('New York', 'New York City'),
  #       ('California', 'San Bernardino'),
  # ('California', 'Orange'),
  # ('California', 'Los Angeles'),
  # ('California', 'Ventura'),
  # ('California', 'Riverside'),

]:
  _df = df_orig.copy()
  _df = _df[(_df.state == state) & (_df.county == county)]
  _df = _df[_df.cases > 30]
  _df['days_since'] = pd.to_datetime(_df.date).sub(_df.date.min()).dt.days.tolist()
  _df['new_daily_cases'] = _df.cases - np.roll(_df.cases, 1)

  # popt, pcov = curve_fit(logistic, _df.days_since.tolist(), _df.cases.tolist(), p0=[1200, 0.2, 22])
  # ax.plot(all_x[1:-1], (logistic(all_x, *popt) - np.roll(logistic(all_x, *popt), 1))[1:-1], linestyle='--')
  # ax.plot(all_x, logistic(all_x, *popt), linestyle='--')

  popt, pcov = curve_fit(generalized_logistic, _df.days_since.tolist(), _df.cases.tolist(), p0=[261, -246, 0.8, 4.3, 0.15, 0.30], maxfev=20000)
  ax.plot(all_x[1:-1], (generalized_logistic(all_x, *popt) - np.roll(generalized_logistic(all_x, *popt), 1))[1:-1], linestyle='--')

  _df[1:].plot(x='days_since', y='new_daily_cases', ax=ax, label=county, logy=False, marker='o', color=ax.get_lines()[-1].get_color())

# Graph apperance
# plt.legend()
plt.xticks(np.arange(0, 45, step=1))
ax.set_xlim(1, 45)
# ax.set_ylim(0, 80)
plt.xlabel("Days since %d cases" % n_since)
plt.ylabel("Daily New Cases")
plt.grid()

fig.savefig("5.png")



fig, ax = plt.subplots(figsize=(18,10))

# Plot growth lines
x = np.arange(_df_us.days_since.max() + 1)
for r in [0.25, 0.35, 0.50]:
  ax.plot(x, n_since * np.power(2, r * x), linestyle='--', color='lightgrey')

for state, county in [
    ('Pennsylvania', 'Philadelphia'),
    ('Pennsylvania', 'Butler'),
    ('Pennsylvania', 'Montgomery'),
    ('Pennsylvania', 'Delaware'),
    ('Pennsylvania', 'Bucks'),
    ('Pennsylvania', 'Chester'),
]:
  _df = df_orig.copy()
  _df = _df[(_df.state == state) & (_df.county == county)]
  _df = _df[_df.cases > n_since]
  _df['days_since'] = pd.to_datetime(_df.date).sub(_df.date.min()).dt.days.tolist()
  _df.plot(x='days_since', y='cases', ax=ax, label=county, logy=True, marker='o', color='red')

for state, county in [
  ('Pennsylvania', 'Allegheny'),
  ('Pennsylvania', 'Beaver'),
  ('Pennsylvania', 'Butler'),
  ('Pennsylvania', 'Westmoreland'),
]:
  _df = df_orig.copy()
  _df = _df[(_df.state == state) & (_df.county == county)]
  _df = _df[_df.cases > n_since]
  _df['days_since'] = pd.to_datetime(_df.date).sub(_df.date.min()).dt.days.tolist()
  _df.plot(x='days_since', y='cases', ax=ax, label=county, logy=True, marker='o', color='blue')

# Graph apperance
plt.legend()
plt.xticks(np.arange(0, 26, step=1))
ax.set_xlim(0, 26)
ax.set_ylim(10, 10000)
plt.xlabel("Days since %d cases" % n_since)
plt.ylabel("Cases")
plt.grid()
fig.savefig("6.png")