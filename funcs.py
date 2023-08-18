import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import json


def event_action(df3):  # меняем текстовые значения таргет-колонки на 1/0 согласно предоставленного списка
    print('event_action start')
    target_action = ['sub_car_claim_click',
                     'sub_car_claim_submit_click',
                     'sub_open_dialog_click',
                     'sub_custom_question_submit_click',
                     'sub_call_number_click',
                     'sub_callback_submit_click',
                     'sub_submit_success',
                     'sub_car_request_submit_click'
                     ]

    df3['event_action'] = df3['event_action'].apply(lambda x: 1 if x in target_action else 0)

    return df3


def sample_df3(df3, total_rows=200000, neg_percent=50, pos_percent=50):
    # функция для обработки открытого DF.
    # позволяет задать итоговый размер датасета и процент значений 0 и 1 в таргет-колонке
    # ТРЕБУЕТ предвариательной обработки датасета функцией event_action()

    # print('sample_df3 start')
    df3_pos = df3[df3['event_action'] == 1].sample(int(total_rows / 100 * pos_percent))
    df3_neg = df3[df3['event_action'] == 0].sample(int(total_rows / 100 * neg_percent))
    df3_pos = df3_pos.reset_index()
    df3_neg = df3_neg.reset_index()
    df3_pos = df3_pos.drop('index', axis=1)
    df3_neg = df3_neg.drop('index', axis=1)
    df3 = pd.concat([df3_pos, df3_neg])

    return df3


def ad_campaign(df3):
    # При первом запуске:
        # 1) создает словарь 'рекламная кампания': процент успеха кампании.
        # Первый запуск рекоммендуется производить на полном датасете
        # 2) Записывает словарь в файл
        # 3) создает в датасете колонку с процентом успеха соответствующей рекламной кампании

    # print('ad_campaign start')
    try:
        with open('data/utm_c_frec_dict2.json', 'r') as f:
            utm_c_frec_dict = json.load(f)
    except FileNotFoundError:
        # print("oh.., looks like its the first time you run it - lil' bit longer then, m8. pls hold:)")
        utm_c_frec_dict = {}

        for pos in df3.utm_campaign.unique():
            if len(df3[(df3.utm_campaign == pos) & (df3.event_action == 1)]) == 0:
                utm_c_frec_dict[str(pos)] = 0
            else:
                utm_c_frec_dict[str(pos)] = round(
                    len(df3[(df3.utm_campaign == pos) & (df3.event_action == 1)]) / len(df3[df3.utm_campaign == pos]),
                    5)

        with open('data/utm_c_frec_dict2.json', 'w') as f:
            json.dump(utm_c_frec_dict, f)

    finally:
        df3['camp_succ_rate'] = df3.utm_campaign.apply(lambda x: utm_c_frec_dict[str(x)])

    return df3


def ad_campaign_v_2(df3):
    # версия 2, оптимизирован механизм получения нужных значений
    # При первом запуске:
        # 1) создает словарь 'рекламная кампания': процент успеха кампании.
        # Первый запуск рекоммендуется производить на полном датасете
        # 2) Записывает словарь в файл
        # 3) создает в датасете колонку с процентом успеха соответствующей рекламной кампании
    # При последующих запусках берет готовый файл со значениями, далее п.3

    # print('ad_campaign v2 start')
    try:
        with open('data/utm_c_frec_dict3.json', 'r') as f:
            utm_c_frec_dict = json.load(f)
    except FileNotFoundError:
        print("oh.., looks like its the first time you run it - lil' bit longer then, m8. pls hold:)")
        df5 = df3[['utm_campaign', 'event_action']]
        succ_camps = df5[df5.event_action == 1].utm_campaign.value_counts(dropna=False)
        all_camps = df5.utm_campaign.value_counts(dropna=False)

        utm_c_frec_dict = {}

        for pos in all_camps.keys():
            if pos in succ_camps.keys():
                utm_c_frec_dict[str(pos)] = round(succ_camps[pos] / all_camps[pos], 5)
            else:
                utm_c_frec_dict[str(pos)] = 0

        with open('data/utm_c_frec_dict3.json', 'w') as f:
            json.dump(utm_c_frec_dict, f)

    finally:
        df3['camp_succ_rate'] = df3.utm_campaign.apply(lambda x: utm_c_frec_dict[str(x)])

    return df3


def day_of_week(df3):
    # Превращает колонку с датой в колонку с днем недели

    # print('day_of_week start')
    df3['new_date'] = pd.to_datetime(df3['visit_date'])
    df3['day_of_week'] = df3.new_date.dt.dayofweek

    df3 = df3.drop('new_date', axis=1)

    return df3


def empties(df3):
    # Заполняет колонки с большим количеством пропусков

    # print('empties end')
    df3.loc[df3.utm_source.isna() == True, 'utm_source'] = 'other'
    df3.loc[df3.utm_adcontent.isna() == True, 'utm_adcontent'] = 'Other'
    df3.loc[df3.device_brand.isna() == True, 'device_brand'] = 'other'

    return df3


def resolution_func(df3):
    # Для каждого из трех типов дейвайсов создает градацию разрешений: высокое/среднее/низкое,
    # определяет принадлежность и записывает в новую колонку device_screen_resolution_engeneered
    # в последствии кодируется ohe

    # print('resolution_func start')
    bounds = []
    df3['resolution'] = df3.device_screen_resolution.apply(lambda x: eval(x.replace('x', '*')))
    for device in df3.device_category.unique():
        q25 = df3[df3.device_category == device].resolution.quantile(0.25)
        q75 = df3[df3.device_category == device].resolution.quantile(0.75)
        iqr = q75 - q25
        bounds.append((device, q25 - 1.5 * iqr, q75 + 1.5 * iqr))

    test_list = list(df3.device_screen_resolution)
    test_list2 = list(df3.device_category)

    for i in range(len(test_list)):
        test_list[i] = eval(test_list[i].replace('x', '*'))

    tst_l = list(zip(test_list2, test_list))

    resolution = []

    for i in range(len(tst_l)):
        if tst_l[i][0] == bounds[0][0]:
            resolution.append(bounds[0][0] + '_high' if tst_l[i][1] >= bounds[0][2] * 0.7 else (
                bounds[0][0] + '_medium' if bounds[0][2] * 0.7 > tst_l[i][1] >= bounds[0][2] * 0.1 else bounds[0][
                                                                                                            0] + '_low'))
        elif tst_l[i][0] == bounds[1][0]:
            resolution.append(bounds[1][0] + '_high' if tst_l[i][1] >= bounds[1][2] * 0.7 else (
                bounds[1][0] + '_medium' if bounds[1][2] * 0.7 > tst_l[i][1] >= bounds[1][2] * 0.1 else bounds[1][
                                                                                                            0] + '_low'))
        elif tst_l[i][0] == bounds[2][0]:
            resolution.append(bounds[2][0] + '_high' if tst_l[i][1] >= bounds[2][2] * 0.7 else (
                bounds[2][0] + '_medium' if bounds[2][2] * 0.7 > tst_l[i][1] >= bounds[2][2] * 0.1 else bounds[2][
                                                                                                            0] + '_low'))

    df3['device_screen_resolution_engeneered'] = resolution
    df3 = df3.drop('resolution', axis=1)

    return df3


def resolution_func_v_2(df3):
    # Пересчитывает разрешение в int значение для последующего скалирования
    # дальше модель сама разберется:)

    # print('resolution_func v2 start')
    df3['device_screen_resolution'] = df3.device_screen_resolution.apply(lambda x: eval(x.replace('x', '*')))

    return df3


def country(df3, trsh=0.001):

    # Создает список стран и количество их появлений в датасете
    # Оставляет только страны колечество появлений которых составляет больше 0.001 от длины датасета
    # к примеру для датасета в 200 000 оставляет страны, мелькнувшие более 200 раз

    # print('country start')
    country_list = list(df3.geo_country.unique())
    for i in range(len(country_list)):
        country_list[i] = (len(df3[df3.geo_country == country_list[i]]), country_list[i])
    country_list = sorted(country_list, reverse=True)

    df3_len = len(df3)
    for item in country_list:
        if item[0] / df3_len >= trsh:
            continue
        else:
            df3.loc[df3.geo_country == item[1], 'geo_country'] = 'some_unimportant_country'

    return df3


def country_v_2(df3):
    # При первом запуске:
        # 1) создает список стран с процентом успеха каждой страны от общего количества значений 1 в таргет-колонке
        # Первый запуск рекоммендуется производить на полном датасете
        # 2) Записывает список в файл
        # 3) создает в датасете колонку с "процентом успеха" соответствующей страны
    # При последующих запусках берет готовый файл со значениями, далее п.3

    # print('country v2  start')
    counter = 0
    country_list_new = dict()

    try:
        with open('data/country_list_new.txt', 'r') as f:
            for line in f:
                line = line.rstrip('\n').replace("%", '')
                tuple_elements = line.split('*')
                my_tuple = (tuple_elements[0], eval(tuple_elements[1]))
                country_list_new[my_tuple[0]] = my_tuple[1]

    except FileNotFoundError:
        # print("oh.., looks like its the first time you run it - lil' bit longer then, m8. pls hold:)")
        succ_total = len(df3[df3.event_action == 1])
        country_list_success = df3[df3.event_action == 1].geo_country.value_counts().sort_values(ascending=False)
        country_list_new = []
        for country in country_list_success.keys():
            country_list_new.append(f'{country}*{str(round(country_list_success[country] / succ_total, 4))}%')
            counter += 1
            if counter == 23:
                break

        with open('data/country_list_new.txt', 'w') as f:
            for t in country_list_new:
                f.write(str(t) + '\n')

        country_list_new = dict()
        with open('data/country_list_new.txt', 'r') as f:
            for line in f:
                line = line.rstrip('\n').replace("%", '')
                tuple_elements = line.split('*')
                my_tuple = (tuple_elements[0], eval(tuple_elements[1]))
                country_list_new[my_tuple[0]] = my_tuple[1]

    finally:
        df3['geo_country_succ_perc'] = df3['geo_country'].apply(
            lambda x: country_list_new[x] if x in country_list_new else 0.0001)

    return df3


def city(df3, trsh=0.001):
    # При первом запуске:
        # 1) создает список .value_counts по городам
        # Первый запуск рекоммендуется производить на полном датасете
        # 2) Записывает список в файл
        # 3)Оставляет только города, колечество появлений которых составляет больше 0.001 от длины полного датасета
    # При последующих запусках берет готовый файл со значениями, далее п.3

    # print('city start')
    city_list = []
    df3_len = len(df3)
    try:
        with open('data/city_list1.txt', 'r') as f:
            for line in f:
                line = line.rstrip('\n').replace('(', '').replace(')', '').replace("'", '')
                tuple_elements = [int(e.strip()) if e.strip().isdigit() else e.strip() for e in line.split(',')]
                my_tuple = tuple(tuple_elements)
                city_list.append(my_tuple)

    except FileNotFoundError:
        # print("oh.., looks like its the first time you run it - lil' bit longer then, m8. pls hold:)")

        city_list = list(zip(df3.geo_city.value_counts().values, df3.geo_city.value_counts().keys()))
        city_list = sorted(city_list, reverse=True)

        with open('data/city_list1.txt', 'w') as f:
            for t in city_list:
                f.write(str(t) + '\n')

    finally:
        city_list_valid = []

        for item in city_list:
            if round(item[0] / 15000000, 4) >= trsh:  # df3_len, 4) >= trsh:
                city_list_valid.append(item[1])

        df3.loc[(~df3['geo_city'].isin(city_list_valid)), 'geo_city'] = 'some_unimportant_city'

    return df3


def city_v_2(df3):
    # При первом запуске:
        # 1) создает список городов с процентом успеха каждого города от общего количества значений 1 в таргет-колонке
        # Первый запуск рекоммендуется производить на полном датасете
        # 2) Записывает список в файл
        # 3) создает в датасете колонку с "процентом успеха" соответствующего города
    # При последующих запусках берет готовый файл со значениями, далее п.3

    # print('city v2  start')
    counter = 0
    city_list_new = dict()

    try:
        with open('data/city_list_new.txt', 'r') as f:
            for line in f:
                # remove newline character and parentheses
                line = line.rstrip('\n').replace("%", '')
                tuple_elements = line.split('*')
                my_tuple = (tuple_elements[0], eval(tuple_elements[1]))
                city_list_new[my_tuple[0]] = my_tuple[1]

    except FileNotFoundError:
        # print("oh.., looks like its the first time you run it - lil' bit longer then, m8. pls hold:)")

        succ_total = len(df3[df3.event_action == 1])
        city_list_success = df3[df3.event_action == 1].geo_city.value_counts().sort_values(ascending=False)
        city_list_new = []
        for city in city_list_success.keys():
            city_list_new.append(f'{city}*{str(round(city_list_success[city] / succ_total, 4))}%')
            counter += 1
            if counter == 26:
                break

        with open('data/city_list_new.txt', 'w') as f:
            for t in city_list_new:
                f.write(str(t) + '\n')

        with open('data/city_list_new.txt', 'r') as f:
            city_list_new = dict()
            for line in f:
                # remove newline character and parentheses
                line = line.rstrip('\n').replace("%", '')
                tuple_elements = line.split('*')
                my_tuple = (tuple_elements[0], eval(tuple_elements[1]))
                city_list_new[my_tuple[0]] = my_tuple[1]

    finally:
        df3['geo_city_succ_perc'] = df3['geo_city'].apply(lambda x: city_list_new[x] if x in city_list_new else 0.0001)

    return df3


def device_brand(df3, trsh=0.0012):
    # При первом запуске:
        # 1) создает список .value_counts по брендам
        # Первый запуск рекоммендуется производить на полном датасете
        # 2) Записывает список в файл
        # 3)Оставляет только бренды, колечество появлений которых составляет больше 0.0012 от длины полного датасета
    # При последующих запусках берет готовый файл со значениями, далее п.3

    # print('device_brand start')
    brand_list = []
    df3_len = len(df3)

    try:
        with open('data/brand_list1.txt', 'r') as f:
            for line in f:
                line = line.rstrip('\n').replace('(', '').replace(')', '').replace("'", '')
                tuple_elements = [int(e.strip()) if e.strip().isdigit() else e.strip() for e in line.split(',')]
                my_tuple = tuple(tuple_elements)
                brand_list.append(my_tuple)

    except FileNotFoundError:
        # print("oh.., looks like its the first time you run it - lil' bit longer then, m8. pls hold:)")

        brand_list = list(zip(df3.device_brand.value_counts().values, df3.device_brand.value_counts().keys()))
        brand_list = sorted(brand_list, reverse=True)

        with open('data/brand_list1.txt', 'w') as f:
            for t in brand_list:
                f.write(str(t) + '\n')

    finally:
        brand_list_valid = []

        for item in brand_list:
            if item[0] / df3_len >= trsh:
                brand_list_valid.append(item[1])

        df3.loc[(~df3['device_brand'].isin(brand_list_valid)), 'device_brand'] = 'some_unimportant_brand'

    return df3


def device_brand_v_2(df3):
    # При первом запуске:
        # 1) создает список брендов с процентом успеха каждого бренда от общего количества значений 1 в таргет-колонке
        # Первый запуск рекоммендуется производить на полном датасете
        # 2) Записывает список в файл
        # 3) создает в датасете колонку с "процентом успеха" соответствующего бренда
    # При последующих запусках берет готовый файл со значениями, далее п.3

    # print('device_brand v2 start')
    counter = 0
    device_brand_list_new = dict()

    try:
        with open('data/device_brand_list_new1.txt', 'r') as f:
            for line in f:
                # remove newline character and parentheses
                line = line.rstrip('\n').replace("%", '')
                tuple_elements = line.split('*')
                my_tuple = (tuple_elements[0], eval(tuple_elements[1]))
                device_brand_list_new[my_tuple[0]] = my_tuple[1]

    except FileNotFoundError:
        # print("oh.., looks like its the first time you run it - lil' bit longer then, m8. pls hold:)")
        succ_total = len(df3[df3.event_action == 1])
        device_brand_list_success = df3[df3.event_action == 1].device_brand.value_counts().sort_values(ascending=False)
        device_brand_list_new = []
        for device_brand in device_brand_list_success.keys():
            device_brand_list_new.append(
                f'{device_brand}*{str(round(device_brand_list_success[device_brand] / succ_total, 4))}%')
            counter += 1
            if counter == 23:
                break

        with open('data/device_brand_list_new1.txt', 'w') as f:
            for t in device_brand_list_new:
                f.write(str(t) + '\n')

        device_brand_list_new = dict()
        with open('data/device_brand_list_new1.txt', 'r') as f:
            for line in f:
                # remove newline character and parentheses
                line = line.rstrip('\n').replace("%", '')
                tuple_elements = line.split('*')
                my_tuple = (tuple_elements[0], eval(tuple_elements[1]))
                device_brand_list_new[my_tuple[0]] = my_tuple[1]

    finally:
        df3['device_brand_succ_perc'] = df3['device_brand'].apply(
            lambda x: device_brand_list_new[x] if x in device_brand_list_new else 0.0001)

    return df3


def encode_stuff(df3):
    # кодирование в обход MemoryErrror, стало ненужно после покупки доп памяти в компьютер

    # print('encode_stuff start')
    cols_to_encode = ['utm_source',
                      'utm_medium',
                      'utm_adcontent',
                      # 'device_brand',
                      'device_category',
                      'device_screen_resolution',
                      'device_browser',
                      'utm_campaign'
                      # ,'geo_country',
                      # 'geo_city'
                      ]

    # encoding

    for col in cols_to_encode:
        pre_encoded_df3 = df3[[col]]
        encoder = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False)
        encoded_array = encoder.fit_transform(pre_encoded_df3)
        feature_names = encoder.get_feature_names_out()
        encoded_df3 = pd.DataFrame(encoded_array, columns=feature_names)

        df3[feature_names] = encoded_df3.values

    df3 = df3.drop(cols_to_encode, axis=1)

    return df3


def scale_stuff(df3):
    # скалировщик
    # scaling
    print('scale_stuff start')
    cols_to_scale = [  # 'visit_number',
        'day_of_week',
        'device_screen_resolution'
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df3.loc[:, cols_to_scale])
    scaled_feature_names = [f'{name}_scaled' for name in scaler.get_feature_names_out()]
    df3[scaled_feature_names] = scaled_features
    df3 = df3.drop(cols_to_scale, axis=1)

    return df3


def filter_stuff(df3):
    # Очистка от лишних колонок
    # + "костыль" для удаления индекс-колонок если они где то образовываются

    # print('filter_stuff start')
    cols_to_drop = [
        'session_id',
        'hit_date',
        'hit_time',
        'hit_number',
        'hit_type',
        'hit_referer',
        'hit_page_path',
        'event_category',
        'event_label',
        'event_value',
        'client_id',
        # 'new_date',
        'visit_date',
        'visit_number',
        'utm_keyword',
        'device_os',
        'device_model',
        'visit_time'
    ]

    cols_to_encode = [
        'utm_source',
        'utm_medium',
        'utm_adcontent',
        'device_brand',
        'device_category',
        'device_screen_resolution',
        'device_browser',
        'utm_campaign',
        'geo_country',
        'geo_city'
    ]

    df3 = df3.drop(cols_to_drop, axis=1)
    # df3 = df3.drop(cols_to_encode, axis=1)

    try:
        df3 = df3.drop('Unnamed: 0', axis=1)
    except KeyError:
        pass
    try:
        df3 = df3.drop('Unnamed: 0.1', axis=1)
    except KeyError:
        pass
    try:
        df3 = df3.drop('Unnamed: 0.2', axis=1)
    except KeyError:
        pass

    return df3


def check_stuff(df3):
    # инструмент отлова ошибок, в финальном пайапе не используется
    # print('check_stuff start')
    counter = 0
    for feature in df3.columns:
        if df3[feature].dtype != 'O':
            # print(feature, ' - ', df3[feature].dtype)
            counter += 1
        else:
            print(feature)
    print(counter == len(df3.columns))

    # checking 2
    counter = 0
    for feature in df3.columns:
        if len(df3[df3[str(feature)].isna() == True]) != 0:
            print(feature, ' - ', len(df3[df3[str(feature)].isna() == True]))
            counter += 1

    if counter == 0:
        print('vse zaebis", pustukh fi4ei net')

    return df3


def check_stuff_2(df3):
    # инструмент отлова ошибок, в финальном пайапе не используется
    # print('check_stuff_2 start')

    counter = 0
    for feature in df3.columns:
        if df3[feature].dtype != 'O':
            # print(feature, ' - ', df3[feature].dtype)
            counter += 1
        else:
            print(feature)
    print(counter == len(df3.columns))

    # checking 2
    empty_features = False
    if sum(df3.isnull().sum()) != 0:
        print(df3.isnull().sum())
        empty_features = True

    if empty_features == False:
        print('vse zaebis", pustukh fi4ei net')

    return df3


def check_stuff_3(df3):
    # инструмент отлова ошибок, в финальном пайапе не используется

    for column in df3.columns:
        print(column)
        print(df3[column].value_counts())
    print(len(df3.columns))

    return df3


def predict_stuff(df3):
    # ручной предикт, ранняя версия, бесполезен
    y = df3['event_action']

    df3 = df3.drop('event_action', axis=1)
    print(df3.columns)
    x_train, x_test, y_train, y_test = train_test_split(df3, y, test_size=0.3)

    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt')
    rf.fit(x_train, y_train)

    predicted_train = rf.predict(x_train)
    predicted_test = rf.predict(x_test)

    print('train acc score - ', accuracy_score(y_train, predicted_train))
    print('test acc score - ', accuracy_score(y_test, predicted_test))

    print('train roc score - ', roc_auc_score(y_train, rf.predict_proba(x_train)[:, 1]))
    print('test roc score - ', roc_auc_score(y_test, rf.predict_proba(x_test)[:, 1]))
    pass
