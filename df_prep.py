import pandas as pd
import funcs
import json

# 1) подготовка датасета для обучения модели
# 2) параллельно создаются необходимые для работы функций файлы

df = pd.read_csv('data/ga_hits.csv')
df2 = pd.read_csv('data/ga_sessions.csv')
df3 = pd.merge(df, df2, on='session_id')
df = None
df2 = None

df3 = funcs.event_action(df3)
print('event action end')
df3 = funcs.pre_filter(df3)
print('filter_stuff end')


feature_list = df3.columns

for feature in feature_list:
    if 'Unnamed' in feature:
        df3 = df3.drop([feature], axis=1)

df4 = funcs.sample_df3(df3, 200000, 50, 50)
print('sampling end')
df4.to_csv('data/df3_200k_50n_50p_2_step.csv', index=False)


for i in range(10):
    sample_row_dict = df3.sample(n=1).to_dict(orient='records')[0]
    with open(f'examples/example_{i}.json', 'w') as file:
        json.dump(sample_row_dict, file)

print('examples end')

df3 = funcs.ad_campaign_v_2(df3)
print('ad camp end')

df3 = funcs.city_v_2(df3)
print('city v2 end')

df3 = funcs.city(df3)
print('city end')

df3 = funcs.country_v_2(df3)
print('country v2 end')

df3 = funcs.country_v_3(df3)
print('country end')

df3 = funcs.device_brand_v_2(df3)
print('device brand v2 end')

df3 = funcs.device_brand(df3)
print('device brand end')
