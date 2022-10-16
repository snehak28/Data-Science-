# Data-Science-
#Data analysis on covid-19 vaccination
Source Code:


#covid 19 data:

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import os
import glob
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
############################# Read Covid_19_india data into the datatframe Covid19India ##################
#####################
print("Read Covid_19_india data into the dataframe Covid19India\n")
Covid19India = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
############################# Read Covid_19_india data into the datatframe Covid19India ##################
#####################
print("Read Covid_19_india data into the dataframe Covid19India\n")
Covid19India = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
df_vaccine=pd.read_csv(r"../input/covid19-in-india/covid_vaccine_statewise.csv


#Plot the curve of Confirmed, Cured and Death cases in India from Feb 2020 till date
print("Plotting the curve of Confirmed, Cured and Death Cases in India from Feb 2020
till date\n\n")
Covid19ByDay = Covid19India.groupby('Date')['Cured', 'Deaths', 'Confirmed'].sum()
Covid19ByDay['CuredPerDay'] = Covid19ByDay.Cured - Covid19ByDay.Cured.shift(1)
Covid19ByDay['DeathsPerDay'] = Covid19ByDay.Deaths - Covid19ByDay.Deaths.shift(1)
Covid19ByDay['ConfirmedPerDay'] = Covid19ByDay.Confirmed - Covid19ByDay.Confirmed.shi
ft(1)
## Calculate Rolling Mean of 7 days
Covid19ByDay['CuredPerDay7MM'] = Covid19ByDay['CuredPerDay'].rolling(window = 7).mean
()
Covid19ByDay['DeathsPerDay7MM'] = Covid19ByDay['DeathsPerDay'].rolling(window = 7).me
an()
Covid19ByDay['ConfirmedPerDay7MM'] = Covid19ByDay['ConfirmedPerDay'].rolling(window =
7).mean()
Covid19ByDay = Covid19ByDay[(Covid19ByDay.index >= '2020-02-01')]
fig,ax = plt.subplots(ncols=1,nrows=3,dpi=100,figsize=(20,18))
ax[0].plot(Covid19ByDay.Confirmed, color = 'dodgerblue')
ax[0].plot(Covid19ByDay.Cured, color = 'green')
ax[0].plot(Covid19ByDay.Deaths, color = 'red')
ax[0].legend(['Confirmed','Cured', 'Deaths'], prop={'size': 20})


ax[0].set_ylabel("Cumulative graph \n In Crores", fontsize=16)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax[0].xaxis.set_major_formatter(DateFormatter('%b%y'))
for label in ax[0].xaxis.get_ticklabels():
label.set_rotation(45)
Xstart, Xend = ax[0].get_xlim()
ax[0].hlines(y=[0.5e+7, 1.0e+7, 1.5e+7], xmin=Xstart, xmax=Xend, color='gray', linew
idth = 0.7)
ax[1].plot(Covid19ByDay.ConfirmedPerDay, color = 'dodgerblue')
ax[1].plot(Covid19ByDay.CuredPerDay, color = 'green')
ax[1].plot(Covid19ByDay.DeathsPerDay, color = 'red')
ax[1].legend(['Confirmed','Cured', 'Deaths'], prop={'size': 20})
ax[1].set_ylabel("Daywise pattern", fontsize =16)
ax[1].tick_params(axis='both', labelsize=16)
ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax[1].xaxis.set_major_formatter(DateFormatter('%b%y'))
for label in ax[1].xaxis.get_ticklabels():
label.set_rotation(45)
Xstart, Xend = ax[1].get_xlim()
ax[1].hlines(y=[100000, 200000, 300000], xmin=Xstart, xmax=Xend, color='gray', linew
idth = 0.7)
ax[2].plot(Covid19ByDay.ConfirmedPerDay7MM, color = 'dodgerblue')
ax[2].plot(Covid19ByDay.CuredPerDay7MM, color = 'green')
ax[2].plot(Covid19ByDay.DeathsPerDay7MM, color = 'red')
ax[2].legend(['Confirmed','Cured', 'Deaths'], prop={'size': 20})
ax[2].set_ylabel('7 Day Rolling Average', fontsize =16)
ax[2].tick_params(axis='both', labelsize=16)
ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax[2].xaxis.set_major_formatter(DateFormatter('%b%y'))
for label in ax[2].xaxis.get_ticklabels():
label.set_rotation(45)
Xstart, Xend = ax[2].get_xlim()
ax[2].hlines(y=[100000, 200000, 300000], xmin=Xstart, xmax=Xend, color='gray', linew
idth = 0.7)
plt.show()

#Worst 5 affected states of India?

### Plot the 5 Worst Affected States
print("What are the worst 5 affected states of India?\n")
LatestDate = Covid19ToDate.reset_index().Date[0]
Covid19ToDate = Covid19ToDate[Covid19ToDate['Date'] == LatestDate].sort_values(by='Confirmed', ascending
=False)
fig,ax = plt.subplots(ncols=3,nrows=1,dpi=100,figsize=(18,8))
CovidWorstStates = Covid19ToDate.head(5)
sns.barplot(data = CovidWorstStates, x='State', y='Confirmed', ax=ax[0])
ax[0].set_yticklabels([5, 10, 15, 20, 25, 30, 35, 40, 45])
ax[0].set_ylabel('Confirmed (in lakhs)', fontsize =15)
ax[0].set_xticklabels(CovidWorstStates.State, rotation=90, fontsize = 14)
ax[0].set_xlabel(" ")
sns.barplot(data = CovidWorstStates, x='State', y='Cured', ax=ax[1])
ax[1].set_yticklabels([5, 10, 15, 20, 25, 30, 35, 40])
ax[1].set_ylabel('Cured (in lakhs)', fontsize =15)
ax[1].set_xticklabels(CovidWorstStates.State, rotation=90, fontsize = 14)
ax[1].set_xlabel(" ")
sns.barplot(data = CovidWorstStates, x='State', y='Deaths', ax=ax[2])
ax[2].set_ylabel('Deaths', fontsize =15)
ax[2].set_xticklabels(CovidWorstStates.State, rotation=90, fontsize = 14)
ax[2].set_xlabel(" ")
##plt.suptitle("Worst Indian States Affected", fontsize=20, fontweight='bold')
plt.show()


# Comparing the worst affected states with rest of India
print("Comparing the worst affected states with rest of India\n")
NumsConfirmed = list(Covid19ToDate['Confirmed'][:5].values)
NumsDeaths = list(Covid19ToDate.sort_values(by='Deaths', ascending = False)['Deaths']
[:5])
labels = list(Covid19ToDate['State'][:5].values)
NumsConfirmed.append(Covid19ToDate['Confirmed'][5:].sum())
NumsDeaths.append(Covid19ToDate.sort_values(by='Deaths', ascending = False)['Deaths']
[5:].sum())
labels.append('Rest of India')
fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (16,16))
ax[0].pie(NumsConfirmed, labels=labels, autopct = '%1.1f%%', explode=[0.05]*len(label
s))
ax[0].set_title('% share of Confirmed Cases', fontsize = 16)
ax[1].pie(NumsDeaths, labels=labels, autopct = '%1.1f%%', explode=[0.05]*len(labels))
ax[1].set_title('% share of Deaths', fontsize = 16)
plt.show()

ax[1].set_title('% share of Deaths', fontsize = 16)
plt.show()






#visualiazation of complete data

# Set the width and height of the figure
plt.figure(figsize=(14,6))
# Add title
plt.title("Full dataset")
# Line chart
sns.lineplot(data=testing_df)
df_state = df_state[:-2]
df_state_trans = df_state.T
df_state_trans = df_state_trans.reset_index(drop=False)
column_names = list(df_state_trans.iloc[0].values)
column_names.remove("State")
column_names = ["Date"] + column_names
df_state_trans.columns = column_names
df_state_trans = df_state_trans.drop(0)
df_state_trans.Date = pd.to_datetime(df_state_trans.Date, format="%d/%m/%Y")
df_state_trans = df_state_trans.set_index("Date")
for name in df_state_trans.select_dtypes("object"):
df_state_trans[name] = df_state_trans[name].astype('int')


#absolute number of vaccines administered

plot_ts(df_state_trans,title="Absolute Number of Vaccines Administered", x_title="Number of doses", y_titl
e="Date", width=1000, height=600)

#Percentage of Population That should have been doubly vaccinated with acquired doses

In [8]:
linkcode
df_state_trans_percent = pd.DataFrame()
for state_name in df_state_trans.columns:
pop = df_pop[df_pop["State"] == state_name]["Population"].values[0]
if pop is not None:
df_state_trans_percent[state_name] = (df_state_trans[state_name]/ (2 * pop))
* 100
plot_ts(df_state_trans_percent,title="Percentage of Population That Should Have Been
Doubly Vaccinated With Acquired Doses", x_title="Percentage of population", y_title="
Date", width=1000, height=600)



#vaccniation over time in rajasthan
# Rajasthan = df_vaccine[df_vaccine["State"]=="Rajasthan"]
fig = px.line(Rajasthan,x="Date",y="Total Vaccinatons",title="Vaccination over time-->Rajasthan")
fig.update_xaxes(rangeslider_visible=True)

#forecasting
df_confirmed_india.columns = ['ds','y']
df_confirmed_india['ds'] = pd.to_datetime(df_confirmed_india['ds'])
m = Prophet()
m.fit(df_confirmed_india)
future = m.make_future_dataframe(periods=21)
future.tail()
forecast = m.predict(future)
forecast.tail()
m.plot(forecast)

#ladakh testing
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Ladakh dataset")
# Line chart
sns.lineplot(data=testing_df_Ladakh)

#total description of datasets
# Set the width and height of the figure
plt.figure(figsize=(14,6))
# Add title
plt.title("Full dataset")
# Line chart
sns.lineplot(data=testing_df)

#west Bengal dataset
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("WestBengal dataset")
# Line chart
sns.lineplot(data=covid19India_df_WestBengal)

#top 10 states with high vaccination rate
########################## ########################## Top 10 states leading in vaccinating its people on p
ercent of population #####################
###################### COMPARED with how much the worst affected States are Vaccinating? #################
##################
print(" Top 10 states leading in vaccinating its people on percent of population \n")
print(" COMPARED with how much the worst affected States are Vaccinating? \n")
fig, ax = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(16,8))
StateVaccineLatest['PopVacc']=(StateVaccineLatest.TotalVaccinated/StateVaccineLatest.Population)*100
StateVaccineLatest = StateVaccineLatest.sort_values(by='PopVacc', ascending=False)
sns.barplot(data=StateVaccineLatest.head(10), x='State', y='PopVacc', ax=ax[0])
ax[0].set_ylabel('% of State Population Vaccinated', fontsize=16)
ax[0].set_xlabel('States', fontsize=16)
ax[0].set_xticklabels(StateVaccineLatest.head(10).State, rotation=90, fontsize = 14)
for bar in ax[0].patches:
ax[0].annotate(format(bar.get_height(), '.1f'),
(bar.get_x() + bar.get_width() / 2,
bar.get_height()), ha='center', va='center',
size=12, xytext=(0, 8),
textcoords='offset points')
data = pd.DataFrame(columns=['State', 'PopVacc'])
for i in range(len(CovidStates)):
data = data.append(StateVaccineLatest[StateVaccineLatest['State'] == CovidStates[i]][['State', 'PopVa
cc']])

sns.barplot(data=data.sort_values(by='PopVacc',ascending=False), x='State', y='PopVacc', ax=ax[1])
ax[1].set_ylabel('% of State Population Vaccinated', fontsize=16)
ax[1].set_xlabel('States', fontsize=20)
ax[1].set_xticklabels(data.sort_values(by='PopVacc',ascending=False).State, rotation=90, fontsize = 14)
for bar in ax[1].patches:
ax[1].annotate(format(bar.get_height(), '.1f'),
(bar.get_x() + bar.get_width() / 2,
bar.get_height()), ha='center', va='center',
size=14, xytext=(0, 8),
textcoords='offset points')


#how India is vaccinating:
########################################## How is India vaccinating ######################################
##################
print("How is India Vaccinating?\n")
fig, ax = plt.subplots(ncols=1, nrows=1, dpi=100, figsize=(8,6))
sns.lineplot(data=CovidVaccine[CovidVaccine['State']=='India'], x='Date', y='TotalVaccinated', ax=ax)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(DateFormatter('%b%y'))
ax.set_ylabel('Total Indians Vaccinated so far in millions', fontsize=14)
ax.set_yticklabels([0,0,20,40,60,80,100,120])
Xstart, Xend = ax.get_xlim()
ax.hlines(y=[20e6, 60e6, 100e6], xmin=Xstart, xmax=Xend, color='gray', linewidth = 0.7)
plt.show()

