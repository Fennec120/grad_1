import pandas as pd
import funcs


df = pd.read_csv('data/ga_hits.csv')
df2 = pd.read_csv('data/ga_sessions.csv')
df3 = pd.merge(df, df2, on='session_id')
df3 = funcs.event_action(df3)
print('event action end')

df3 = funcs.ad_campaign_v_2(df3)
print('ad camp end')

df3 = funcs.city_v_2(df3)
print('city v2 end')

df3 = funcs.city(df3)
print('city end')

df3 = funcs.country_v_2(df3)
print('country v2 end')

df3 = funcs.country(df3)
print('country end')

df3 = funcs.device_brand_v_2(df3)
print('device brand v2 end')

df3 = funcs.device_brand(df3)
print('device brand end')



df3 = pd.merge(df, df2, on='session_id')
df3 = funcs.event_action(df3)

print('event action 2nd go around end')

df3 = funcs.sample_df3(df3, 200000, 50, 50)
print('sampling end')

feature_list = df3.columns
for feature in feature_list:
    if 'Unnamed' in feature:
        df3 = df3.drop([feature], axis=1)

df3.to_csv('data/df3_200k_50n_50p_tst.csv')