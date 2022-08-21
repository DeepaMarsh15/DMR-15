#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install plotly')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df=pd.read_csv('covid_19_clean_complete.csv',parse_dates=["Date"])


# In[6]:


df.head()


# In[7]:


df.rename(columns={'Province/State':'state','Country/Region':'country','Lat':'lat','Long':'long','Date':'date',
                   'Confirmed':'confirmed','Deaths':'deaths','Recovered':'recovered'},inplace=True)


# In[8]:


df.head()


# In[9]:


df['active']=df['confirmed']-df['deaths']-df['recovered']
df.head()


# In[10]:


top=df[df['date']==df['date'].max()]


# In[11]:


world=top.groupby('country')['confirmed','active','deaths'].sum().reset_index()


# In[12]:


world.head()


# In[13]:


figure=px.choropleth(world,locations='country',
                     locationmode='country names',color='active',
                     hover_name='country',range_color=[1,1000],
                     color_continuous_scale='viridis',
                     title='countries with active cases')
figure.show()


# In[14]:


total_cases=df.groupby('date')['date','confirmed'].sum().reset_index()


# In[15]:


total_cases.head()


# In[16]:


plt.figure(figsize=(10,8))
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=15)
plt.xlabel('dates',fontsize=30)
plt.ylabel('confirmed cases')
plt.title('total confirmed cases',fontsize=30)
ax=sns.pointplot(x=total_cases.date.dt.date,y=total_cases.confirmed,color='r')


# In[17]:


top_active=top.groupby('country')['active'].sum().sort_values(ascending=False).head(20).reset_index()


# In[18]:


top_active


# In[19]:


plt.figure(figsize=(15,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('total cases',fontsize=30)
plt.ylabel('country',fontsize=30)
plt.title('top 20 countries having most active cases',fontsize=30)
ax=sns.barplot(x=top_active.active,y=top_active.country)


# In[20]:


china=df[df.country=='China']
china


# In[21]:


china=china.groupby('date')['recovered','deaths','confirmed','active'].sum().reset_index()
china.head()


# In[22]:


us=df[df.country=='US']
us


# In[23]:


us=us.groupby('date')['recovered','deaths','confirmed','active'].sum().reset_index()
us


# In[24]:


us=us.iloc[33:].reset_index().drop('index',axis=1)
us.head()


# In[25]:


italy=df[df.country=='Italy']


# In[26]:


italy=italy.groupby('date')['recovered','deaths','confirmed','active'].sum().reset_index()
italy.head(20)


# In[27]:


italy=italy.iloc[9:].reset_index().drop('index',axis=1)
italy.head()


# In[28]:


india=df[df.country=='India']


# In[29]:


india


# In[30]:


india=india.groupby('date')['recovered','deaths','confirmed','active'].sum().reset_index()
india.head(10)


# In[31]:


india=india.iloc[8:].reset_index().drop('index',axis=1)
india.head()


# In[32]:


plt.figure(figsize=(15,10))
sns.pointplot(china.index,china.confirmed,color='red')
sns.pointplot(us.index,us.confirmed,color='green')
sns.pointplot(italy.index,italy.confirmed,color='blue')
sns.pointplot(india.index,india.confirmed,color='orange')
plt.title=('confirmed cases')
plt.xlabel=('number of days')
plt.ylabel=('confirmed cases')
plt.show()


# In[33]:


plt.figure(figsize=(15,10))
sns.pointplot(china.index,china.recovered,color='red')
sns.pointplot(us.index,us.recovered,color='green')
sns.pointplot(italy.index,italy.recovered,color='blue')
sns.pointplot(india.index,india.recovered,color='orange')
plt.title=('recovered cases')
plt.xlabel=('number of days')
plt.ylabel=('recovered cases')
plt.show()


# In[34]:


plt.figure(figsize=(15,10))
sns.pointplot(china.index,china.deaths,color='red')
sns.pointplot(us.index,us.deaths,color='green')
sns.pointplot(italy.index,italy.deaths,color='blue')
sns.pointplot(india.index,india.deaths,color='orange')
plt.title=('deaths cases')
plt.xlabel=('number of days')
plt.ylabel=('deaths cases')
plt.show()

############India and its States##############
# In[35]:


df_india=pd.read_excel('covid_19_india.xlsx')


# In[36]:


df_india.head()


# In[43]:


df_india['Total Cases']=df_india['Total Confirmed cases ( Foreign National )']+df_india['Total Confirmed cases (Indian National)']


# In[45]:


df_india['Total Active']=df_india['Total Cases']-(df_india['Death']-df_india['Cured'])
total_active=df_india['Total Active'].sum()
print('Total Number of Active COVID 19 cases across India',total_active)
Tot_Cases=df_india.groupby('Name of State / UT')['Total Active'].sum().sort_values(ascending=False).to_frame()
Tot_Cases.style.background_gradient(cmap='hot_r')


# In[46]:


f,ax=plt.subplots(figsize=(12,8))
data=df_india[['Name of State / UT','Total Cases','Cured','Death']]
data.sort_values('Total Cases',ascending=False,inplace=True)
sns.set_color_codes('pastel')
sns.barplot(x='Total Cases',y='Name of State / UT',data=data, label='Total',color='r')
sns.set_color_codes('muted')
sns.barplot(x='Cured',y='Name of State / UT',data=data, label='Cured',color='g')
ax.legend(ncol=2,loc='lower right',frameon=True)
ax.set(ylabel='States and UT', xlabel='Cases')


# In[47]:


dbd_india=pd.read_excel('per_day_cases.xlsx',parse_dates=True,sheet_name='India')
dbd_india


# In[49]:


fig= go.Figure()
fig.add_trace(go.Scatter(x=dbd_india['Date'], y=dbd_india['Total Cases'], mode='lines+markers', name='Total Cases'))
fig.update_layout(title_text='Trend of Coronavirus cases in India(Cumulative Cases)', plot_bgcolor='rgb(230, 230, 230)')
fig.show()


# In[50]:


fig =px.bar(dbd_india, x='Date', y='New Cases', barmode='group', height=400)
fig.update_layout(title_text='Coronavirus cases in India on a daily basis', plot_bgcolor= 'rgb(230, 230, 230)')
fig.show()


# In[51]:


df_confirmed=pd.read_csv('time_series_covid19_confirmed_global.csv')
df_recovered=pd.read_csv('time_series_covid19_recovered_global.csv')
df_deaths=pd.read_csv('time_series_covid19_deaths_global.csv')
df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)


# In[52]:


df_confirmed.head()


# In[53]:


df_recovered.head()


# In[54]:


df_deaths.head()


# In[55]:


df.head()


# In[58]:


df_india_cases=df.query('country == "India"').groupby('date')[['confirmed', 'deaths', 'recovered']].sum().reset_index()
india_confirmed, india_deaths, india_recovered = df_india_cases[['date','confirmed']], df_india_cases[['date','deaths']],df_india_cases[['date','recovered']]
df_india_cases


# In[59]:


df.groupby('date').sum().head()


# In[60]:


confirmed= df.groupby('date').sum()['confirmed'].reset_index()
deaths= df.groupby('date').sum()['deaths'].reset_index()
recovered= df.groupby('date').sum()['recovered'].reset_index()


# In[61]:


fig= go.Figure()
fig.add_trace(go.Scatter(x=confirmed['date'], y=confirmed['confirmed'], mode='lines+markers', name='confirmed', line = dict(color= 'blue')))
fig.add_trace(go.Scatter(x=deaths['date'], y=deaths['deaths'], mode='lines+markers', name='deaths', line = dict(color= 'red')))
fig.add_trace(go.Scatter(x=recovered['date'], y=recovered['recovered'], mode='lines+markers', name='recovered', line = dict(color= 'green')))
fig.update_layout(title_text='Worldwide COVID-19 Cases', xaxis_tickfont_size = 14, yaxis=dict(title='Number of Cases'), plot_bgcolor= 'rgb(230, 230, 230)')
fig.show()

