#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# ## Why do this step?
# 
# - The data we scraped from the UFC website is raw data. Every row contains information about a fight that took place i.e. details like how many strikes were thrown in that fight and who won the fight.
# - To prepare the data for prediction, every row can contain only an accurate representation of what each fighter has done in fights up until that fight! No data that was recorded during the fight can be present in that row.
# - Our Target variable is Winner. The task has to be to predict the winner from the data available of each fighter up until the fight

# ## Looking at the data

# In[1]:


import pandas as pd
import numpy as np
import math

df = pd.read_csv('../../data/total_fight_data.csv', sep=',')
fighter_details = pd.read_csv('../../data/fighter_details.csv')


# In[2]:


df.head()


# In[3]:


df.T


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


df.columns


# ### Column definitions:
# 
# - `R_` and `B_` prefix signifies red and blue corner fighter stats respectively
# - `KD` is number of knockdowns
# - `SIG_STR` is no. of significant strikes 'landed of attempted'
# - `SIG_STR_pct` is significant strikes percentage
# - `TOTAL_STR` is total strikes 'landed of attempted'
# - `TD` is no. of takedowns
# - `TD_pct` is takedown percentages
# - `SUB_ATT` is no. of submission attempts
# - `PASS` is no. times the guard was passed?
# - `REV` is the no. of reversals
# - `CTRL` is the time spent with ground control
# - `HEAD` is no. of significant strinks to the head 'landed of attempted'
# - `BODY` is no. of significant strikes to the body 'landed of attempted'
# - `CLINCH` is no. of significant strikes in the clinch 'landed of attempted'
# - `GROUND` is no. of significant strikes on the ground 'landed of attempted'
# - `win_by` is method of win
# - `last_round` is last round of the fight (ex. if it was a KO in 1st, then this will be 1)
# - `last_round_time` is when the fight ended in the last round
# - `Format` is the format of the fight (3 rounds, 5 rounds etc.)
# - `Referee` is the name of the Ref
# - `date` is the date of the fight
# - `location` is the location in which the event took place
# - `Fight_type` is which weight class and whether it's a title bout or not
# - `Winner` is the winner of the fight
# 
# #### Per fighter career wide stats
# - `SLpM` - Significant Strikes Landed per Minute
# - `Str_Acc.` - Significant Striking Accuracy
# - `SApM` - Significant Strikes Absorbed per Minute
# - `Str_Def` - Significant Strike Defence (the % of opponents strikes that did not land)
# - `TD_Avg` - Average Takedowns Landed per 15 minutes
# - `TD_Acc` - Takedown Accuracy
# - `TD_Def` - Takedown Defense (the % of opponents TD attempts that did not land)
# - `Sub_Avg` - Average Submissions Attempted per 15 minutes 

# ## Todo:
# 
# - Separate `landed of attempted` to separate columns
# 
# - Convert `Fight_type` into two separate columns, `weight_class` and `Title_fight` (True or False)
# 
# - Convert `last_round_time` to `total_time_fought` by using `last_round` and `Format`
# 
# - Convert `CTRL` to `time_in_CTRL`
# 
# - Convert percentages to fractions
# 
# - Since the data is a description of each fight, we have to convert it into a format that shows the compilation data of each fighter up until that fight. This means every row will look a lot different than it looks now.
# 
# - Create `current_win_streak`, `current_lose_streak`, `longest_win_streak`, `wins`, `losses`, `draw`
# 
# - Create fighter `height`, `reach`, `weight`, `age`

# In[7]:





# ### Splitting landed of attempted to different columns

# In[8]:
print(df.columns)
print(df['R_BODY'])
columns = ['R_SIG_STR.', 'B_SIG_STR.', 'R_TOTAL_STR.', 'B_TOTAL_STR.',
       'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD', 'R_BODY','B_BODY', 'R_LEG', 'B_LEG', 
        'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH','B_CLINCH', 'R_GROUND', 'B_GROUND']

# for column in columns:
#     print(f"{column} data type is: {df[column].dtype}")


# In[8]:


attempt_suffix = '_att'
landed_suffix = '_landed'

for column in columns:
    df[column+attempt_suffix] = df[column].apply(lambda X: int(X.split('of')[1]))
    df[column+landed_suffix] = df[column].apply(lambda X: int(X.split('of')[0]))
    
df.drop(columns, axis=1, inplace=True)


# In[9]:


df.columns


# ### Replacing Winner NaNs as Draw

# In[10]:


for column in df.columns:
    if df[column].isnull().sum() != 0:
        print(f"NaN values in {column} = {df[column].isnull().sum()}")


# * 83 missing values in winner and 23 missing values in Referee

# In[11]:


df[df['Winner'].isnull()]['win_by'].value_counts()


# * Here, Overturned means due to drug test being positive and Could not Continue means there was an illegal blow which was not enough to be disqualified but the fighter could not continue.
# * The rest are different forms of draw
# 
# * Replacing all of these with draw

# In[12]:


df['Winner'].fillna('Draw', inplace=True)


# ### Converting percentages to fractions

# In[13]:


pct_columns = ['R_SIG_STR_pct','B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']

def pct_to_frac(X):
    if X != '---':
        return float(X.replace('%', ''))/100
    else:
        # if '---' means it's taking pct of `0 of 0`. 
        # Taking a call here to consider 0 landed of 0 attempted as 0 percentage
        return 0

for column in pct_columns:
    df[column] = df[column].apply(pct_to_frac)


# In[14]:


df.T


# ### Creating a title_bout feature and weight_class

# In[15]:


df['Fight_type'].value_counts()


# In[16]:


df['Fight_type'].value_counts()[df['Fight_type'].value_counts() > 1].index


# In[17]:


df['title_bout'] = df['Fight_type'].apply(lambda X: True if 'Title Bout' in X else False)


# In[18]:


def make_weight_class(X):
    for weight_class in weight_classes:
        if weight_class in X:
            return weight_class
    if X == 'Catch Weight Bout' or 'Catchweight Bout':
        return 'Catch Weight'
    else:
        return 'Open Weight'


# In[19]:


weight_classes = ['Women\'s Strawweight', 'Women\'s Bantamweight', 
                  'Women\'s Featherweight', 'Women\'s Flyweight', 'Lightweight', 
                  'Welterweight', 'Middleweight','Light Heavyweight', 
                  'Heavyweight', 'Featherweight','Bantamweight', 'Flyweight', 'Open Weight']

df['weight_class'] = df['Fight_type'].apply(make_weight_class)


# In[20]:


df[df['weight_class'].isnull()]['Fight_type'].value_counts()


# ### Creating total_time_fought

# In[21]:


df['Format'].value_counts()


# In[22]:


df['Format'].value_counts().index


# In[23]:


time_in_first_round = {'3 Rnd (5-5-5)': 5*60, '5 Rnd (5-5-5-5-5)': 5*60, '1 Rnd + OT (12-3)': 12*60,
       'No Time Limit': 1, '3 Rnd + OT (5-5-5-5)': 5*60, '1 Rnd (20)': 1*20,
       '2 Rnd (5-5)': 5*60, '1 Rnd (15)': 15*60, '1 Rnd (10)': 10*60,
       '1 Rnd (12)':12*60, '1 Rnd + OT (30-5)': 30*60, '1 Rnd (18)': 18*60, '1 Rnd + OT (15-3)': 15*60,
       '1 Rnd (30)': 30*60, '1 Rnd + OT (31-5)': 31*5,
       '1 Rnd + OT (27-3)': 27*60, '1 Rnd + OT (30-3)': 30*60}

exception_format_time = {'1 Rnd + 2OT (15-3-3)': [15*60, 3*60], '1 Rnd + 2OT (24-3-3)': [24*60, 3*60]}

# '1 Rnd + 2OT (15-3-3)' and '1 Rnd + 2OT (24-3-3)' is not included because it has 3 uneven timed rounds. 
# We'll have to deal with it separately


# In[24]:


# Converting to seconds
df['last_round_time'] = df['last_round_time'].apply(lambda X: int(X.split(':')[0])*60 + int(X.split(':')[1]))


# In[25]:


def get_total_time(row):
    if row['Format'] in time_in_first_round.keys():
        return (row['last_round'] - 1) * time_in_first_round[row['Format']] + row['last_round_time']
    elif row['Format'] in exception_format_time.keys():
        if (row['last_round'] - 1) >= 2:
            return exception_format_time[row['Format']][0] + (row['last_round'] - 2) *                     exception_format_time[row['Format']][1] + row['last_round_time']
        else:
            return (row['last_round'] - 1) * exception_format_time[row['Format']][0] + row['last_round_time']
    
# So if the fight ended in round 1, we only need last_round_time. 
# If it ended in round 2, we need the full time of round 1 and the last_round_time
# This works for fights with same time in each round and fights with only two rounds.


# In[26]:


df['total_time_fought(seconds)'] = df.apply(get_total_time, axis=1)


# In[27]:


def get_no_of_rounds(X):
    if X == 'No Time Limit':
        return 1
    else:
        return len(X.split('(')[1].replace(')', '').split('-'))

df['no_of_rounds'] = df['Format'].apply(get_no_of_rounds)


# In[28]:


df


# In[29]:


df.drop(['Format', 'Fight_type', 'last_round_time'], axis = 1, inplace=True)


# ### Create CTRL_time(seconds)

# In[30]:


CTRL_columns = ['R_CTRL','B_CTRL']

def conv_to_sec(X):
    if X != '--':
        return int(X.split(':')[0])*60 + int(X.split(':')[1])
    else:
        # if '--' means there was no time spent on the ground. 
        # Taking a call here to consider this as 0 seconds
        return 0

for column in CTRL_columns:
    df[column+'_time(seconds)'] = df[column].apply(conv_to_sec)


# In[31]:


df.T


# In[32]:


df.drop(['R_CTRL','B_CTRL'], axis = 1, inplace=True)


# ### Create another DataFrame to save the compiled data per fighter (Our Prediction DataFrame)

# In[33]:


df.columns


# In[34]:


df2 = df.copy()


# In[35]:


df2.drop(['R_KD', 'B_KD', 'R_SIG_STR_pct',
       'B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct', 'R_SUB_ATT', 'B_SUB_ATT',
       'R_CTRL_time(seconds)', 'B_CTRL_time(seconds)', 'R_REV', 'B_REV', 'win_by', 'last_round', 
        'R_SIG_STR._att', 'R_SIG_STR._landed',
       'B_SIG_STR._att', 'B_SIG_STR._landed', 'R_TOTAL_STR._att',
       'R_TOTAL_STR._landed', 'B_TOTAL_STR._att', 'B_TOTAL_STR._landed',
       'R_TD_att', 'R_TD_landed', 'B_TD_att', 'B_TD_landed', 'R_HEAD_att',
       'R_HEAD_landed', 'B_HEAD_att', 'B_HEAD_landed', 'R_BODY_att',
       'R_BODY_landed', 'B_BODY_att', 'B_BODY_landed', 'R_LEG_att',
       'R_LEG_landed', 'B_LEG_att', 'B_LEG_landed', 'R_DISTANCE_att',
       'R_DISTANCE_landed', 'B_DISTANCE_att', 'B_DISTANCE_landed',
       'R_CLINCH_att', 'R_CLINCH_landed', 'B_CLINCH_att', 'B_CLINCH_landed',
       'R_GROUND_att', 'R_GROUND_landed', 'B_GROUND_att', 'B_GROUND_landed',
        'total_time_fought(seconds)'], axis = 1, inplace=True)
df2


# ### Compiling Data per fighter

# In[36]:


red_fighters = df['R_fighter'].value_counts().index
blue_fighters = df['B_fighter'].value_counts().index

fighters = list(set(red_fighters) | set(blue_fighters))


# In[37]:


def get_renamed_winner(row):
    if row['R_fighter'] == row['Winner']:
        return 'Red'
    elif row['B_fighter'] == row['Winner']:
        return 'Blue'
    elif row['Winner'] == 'Draw':
        return 'Draw'

df2['Winner'] = df2[['R_fighter', 'B_fighter', 'Winner']].apply(get_renamed_winner, axis=1)


# In[38]:


df = pd.concat([df,pd.get_dummies(df['win_by'], prefix='win_by')],axis=1)
df.drop(['win_by'],axis=1, inplace=True)


# In[39]:


Numerical_columns = ['hero_KD', 'opp_KD', 'hero_SIG_STR_pct',
       'opp_SIG_STR_pct', 'hero_TD_pct', 'opp_TD_pct', 'hero_SUB_ATT', 'opp_SUB_ATT',
        'hero_REV', 'opp_REV', 'hero_SIG_STR._att', 'hero_SIG_STR._landed',
       'opp_SIG_STR._att', 'opp_SIG_STR._landed', 'hero_TOTAL_STR._att',
       'hero_TOTAL_STR._landed', 'opp_TOTAL_STR._att', 'opp_TOTAL_STR._landed',
       'hero_TD_att', 'hero_TD_landed', 'opp_TD_att', 'opp_TD_landed', 'hero_HEAD_att',
       'hero_HEAD_landed', 'opp_HEAD_att', 'opp_HEAD_landed', 'hero_BODY_att',
       'hero_BODY_landed', 'opp_BODY_att', 'opp_BODY_landed', 'hero_LEG_att',
       'hero_LEG_landed', 'opp_LEG_att', 'opp_LEG_landed', 'hero_DISTANCE_att',
       'hero_DISTANCE_landed', 'opp_DISTANCE_att', 'opp_DISTANCE_landed',
       'hero_CLINCH_att', 'hero_CLINCH_landed', 'opp_CLINCH_att', 'opp_CLINCH_landed',
       'hero_GROUND_att', 'hero_GROUND_landed', 'opp_GROUND_att', 'opp_GROUND_landed', 
        'hero_CTRL_time(seconds)', 'opp_CTRL_time(seconds)',
       'total_time_fought(seconds)']

Categorical_columns = ['win_by', 'last_round',
        'Winner', 'title_bout']


# For all `Numerical_columns`, we take the average of those columns for every fighter of every fight they had up until that point.
# 
# For `Categorical_columns`, we have to come up with different ideas for each column:
# 
# * Each `win_by` will be a column of it's own
# * from `last_round` we can get, `total_rounds_fought`
# * from `total_time_fought` we can get `average_time_fought`
# * from `Winner` we get `wins`, `losses`, `draw`, `current_streak`, `longest_streak`
# * from `title_bout` we can get `no_of_title_fights`

# In[40]:


import re

def lreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
    """
    return re.sub('^%s' % pattern, sub, string)


# In[41]:


red = df.groupby('R_fighter')
blue = df.groupby('B_fighter')


# In[42]:


def get_fighter_red(fighter_name):
    try:
        fighter_red = red.get_group(fighter_name)
    except:
        return None
    rename_columns = {}
    for column in fighter_red.columns:
        if re.search('^R_', column) is not None:
            rename_columns[column] = lreplace('R_', 'hero_', column)
        elif re.search('^B_', column) is not None:
            rename_columns[column] = lreplace('B_', 'opp_', column)
    fighter_red = fighter_red.rename(rename_columns, axis='columns')
    return fighter_red


# In[43]:


def get_fighter_blue(fighter_name):
    try:
        fighter_blue = blue.get_group(fighter_name)
    except:
        return None
    rename_columns = {}
    for column in fighter_blue.columns:
        if re.search('^B_', column) is not None:
            rename_columns[column] = lreplace('B_', 'hero_', column)
        elif re.search('^R_', column) is not None:
            rename_columns[column] = lreplace('R_', 'opp_', column)
    fighter_blue = fighter_blue.rename(rename_columns, axis='columns')
    return fighter_blue


# In[44]:


def get_result_stats(result_list):
    result_list.reverse() # To get it in ascending order
    current_win_streak = 0
    current_lose_streak = 0
    longest_win_streak = 0
    wins = 0
    losses = 0
    draw = 0
    for result in result_list:
        if result == 'hero':
            wins += 1
            current_win_streak += 1
            current_lose_streak = 0
            if longest_win_streak < current_win_streak:
                longest_win_streak += 1
        elif result == 'opp':
            losses += 1
            current_win_streak = 0
            current_lose_streak += 1
        elif result == 'draw':
            draw += 1
            current_lose_streak = 0
            current_win_streak = 0
            
    return current_win_streak, current_lose_streak, longest_win_streak, wins, losses, draw


# In[45]:


win_by_columns = ['win_by_Decision - Majority', 'win_by_Decision - Split',
       'win_by_Decision - Unanimous', 'win_by_KO/TKO','win_by_Submission',
       'win_by_TKO - Doctor\'s Stoppage']


# In[46]:


temp_blue_frame = pd.DataFrame()
temp_red_frame = pd.DataFrame()
result_stats = ['current_win_streak', 'current_lose_streak', 'longest_win_streak', 'wins', 'losses', 'draw']

for fighter_name in fighters:
    fighter_red = get_fighter_red(fighter_name)
    fighter_blue = get_fighter_blue(fighter_name)
    fighter_index = None
    
    if fighter_red is None:
        fighter = fighter_blue
        fighter_index = 'blue'
    elif fighter_blue is None:
        fighter = fighter_red
        fighter_index = 'red'
    else:
        fighter = pd.concat([fighter_red, fighter_blue]).sort_index()
    
    fighter['Winner'] = fighter['Winner'].apply(lambda X: 'hero' if X == fighter_name else 'opp')

    for i, index in enumerate(fighter.index):
        fighter_slice = fighter[(i+1):].sort_index(ascending=False)
        s = fighter_slice[Numerical_columns].ewm(span=3, adjust=False).mean().tail(1)
        if len(s) != 0:
            pass
        else:
            s.loc[len(s)] = [np.NaN for _ in s.columns]
        s['total_rounds_fought'] = fighter_slice['last_round'].sum()
        s['total_title_bouts'] = fighter_slice[fighter_slice['title_bout']==True]['title_bout'].count()
        s['hero_fighter'] = fighter_name
        results = get_result_stats(list(fighter_slice['Winner']))
        for result_stat, result in zip(result_stats, results):
            s[result_stat] = result
        win_by_results = fighter_slice[fighter_slice['Winner'] == 'hero'][win_by_columns].sum()
        for win_by_column,win_by_result in zip(win_by_columns, win_by_results):
            s[win_by_column] = win_by_result
        s.index = [index]


        if fighter_index is None:
            if index in fighter_blue.index:
                temp_blue_frame = temp_blue_frame.append(s)
            elif index in fighter_red.index:
                temp_red_frame = temp_red_frame.append(s)
        elif fighter_index == 'blue':
            temp_blue_frame = temp_blue_frame.append(s)
        elif fighter_index == 'red':
            temp_red_frame = temp_red_frame.append(s)


# In[47]:


temp_blue_frame.T


# ### Adding fighter details like height, weight, reach, stance and dob

# In[48]:


fighter_details = fighter_details[fighter_details.index.isin(fighters)]
for col in fighter_details.columns:
    print(f"Number of NaN in {col} : {fighter_details[col].isnull().sum()}")


# In[49]:


def convert_to_cms(X):
    if X is np.NaN:
        return X
    elif len(X.split("'")) == 2:
        feet = float(X.split("'")[0])
        inches = int(X.split("'")[1].replace(' ', '').replace('"',''))
        return (feet * 30.48) + (inches * 2.54)
    else:
        return float(X.replace('"','')) * 2.54


# In[50]:


fighter_details['Height_cms'] = fighter_details['Height'].apply(convert_to_cms)
fighter_details['Reach_cms'] = fighter_details['Reach'].apply(convert_to_cms)


# In[51]:


fighter_details['Weight_lbs'] = fighter_details['Weight'].apply(lambda X: float(X.replace(' lbs.', '')) if X is not np.NaN else X)


# In[52]:


pct_columns = ['Str_Acc','Str_Def', 'TD_Acc', 'TD_Def']

def pct_to_frac(X):
    if X != np.NaN:
        return float(X.replace('%', ''))/100
    else:
        return 0

for column in pct_columns:
    fighter_details[column] = fighter_details[column].apply(pct_to_frac)


# In[53]:


fighter_details.drop(['Height', 'Weight', 'Reach'], axis=1, inplace=True)

fighter_details.drop(
    columns=["SLpM",
            "Str_Acc",
            "SApM",
            "Str_Def",
            "TD_Avg",
            "TD_Acc",
            "TD_Def",
            "Sub_Avg",
        ], inplace=True)

# In[54]:


fighter_details.reset_index(inplace=True)
temp_red_frame.reset_index(inplace=True)
temp_blue_frame.reset_index(inplace=True)


# In[55]:


temp_blue_frame = temp_blue_frame.merge(fighter_details, left_on='hero_fighter', right_on='fighter_name', how='left')
temp_blue_frame.set_index('index', inplace=True)


# In[56]:


temp_blue_frame[['hero_fighter', 'fighter_name', 'Height_cms', 'Weight_lbs', 'DOB']].head(20)


# In[57]:


temp_red_frame = temp_red_frame.merge(fighter_details, left_on='hero_fighter', right_on='fighter_name', how='left')
temp_red_frame.set_index('index', inplace=True)


# In[58]:


temp_blue_frame.drop('fighter_name', axis=1, inplace=True)
temp_red_frame.drop('fighter_name', axis=1, inplace=True)


# In[59]:


blue_frame = temp_blue_frame.add_prefix('B_')
red_frame = temp_red_frame.add_prefix('R_')


# In[60]:


frame = blue_frame.join(red_frame, how='outer')


# In[61]:


rename_cols = {}
for col in frame.columns:
    if 'hero' in col:
        rename_cols[col] = col.replace('_hero_', '_avg_').replace('.', '')
    if 'opp' in col:
        rename_cols[col] = col.replace('_opp_', '_avg_opp_').replace('.', '')
    if 'win_by' in col:
        rename_cols[col] = col.replace(' ', '').replace('-', '_').replace('\'s', '_')


# In[62]:


frame.rename(rename_cols, axis='columns', inplace=True)


# In[63]:


frame.drop(['R_avg_fighter','B_avg_fighter'], axis=1, inplace=True)


# In[64]:


df2 = df2.join(frame, how='outer')


# ### Create Age

# In[65]:


df2['R_DOB'] = pd.to_datetime(df2['R_DOB'])
df2['B_DOB'] = pd.to_datetime(df2['B_DOB'])
df2['date'] = pd.to_datetime(df2['date'])


# In[66]:


def get_age(row):
    B_age = (row['date'] - row['B_DOB']).days
    R_age = (row['date'] - row['R_DOB']).days
    if np.isnan(B_age)!=True:
        B_age = math.floor(B_age/365.25)
    if np.isnan(R_age)!=True:
        R_age = math.floor(R_age/365.25)
    return pd.Series([B_age, R_age], index=['B_age', 'R_age'])


# In[67]:


df2[['B_age', 'R_age']]= df2[['date', 'R_DOB', 'B_DOB']].apply(get_age, axis=1)


# In[68]:


df2.drop(['R_DOB', 'B_DOB'], axis=1, inplace=True)


# In[69]:


# df2.drop(df2.index[df2['Winner'] == 'draw'], inplace = True)


# In[70]:


df2.to_csv('../../data/infdata.csv', index=False)


# In[71]:


df2


# In[ ]:




