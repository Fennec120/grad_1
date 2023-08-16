import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import json
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import funcs


cv_scores = []

df4 = pd.read_csv('data/df3_200k_50n_50p_tst.csv')

categorical_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

f_transformer = ColumnTransformer(transformers=[
    ('categorical', categorical_transformer, make_column_selector(dtype_include='object'))
])

preprocessor = Pipeline(steps=[
    # ('event', FunctionTransformer(event_action)),
    # ('sampling', FunctionTransformer(sample_df3)),
    ('ad_campaign_feature_creating', FunctionTransformer(funcs.ad_campaign_v_2)),
    ('day_of_week', FunctionTransformer(funcs.day_of_week)),
    ('empties', FunctionTransformer(funcs.empties)),
    # ('resolution_func', FunctionTransformer(resolution_func)),
    ('resolution_func v2', FunctionTransformer(funcs.resolution_func_v_2)),
    ('country v2', FunctionTransformer(funcs.country_v_2)),
    ('country', FunctionTransformer(funcs.country)),
    ('city v2', FunctionTransformer(funcs.city_v_2)),
    ('city', FunctionTransformer(funcs.city)),
    ('device brand v2', FunctionTransformer(funcs.device_brand_v_2)),
    ('device brand', FunctionTransformer(funcs.device_brand)),
    ('filter_stuff', FunctionTransformer(funcs.filter_stuff)),
    # ('encode_stuff', FunctionTransformer(encode_stuff)),
    ('scale_stuff(resol, day_of_week )', FunctionTransformer(funcs.scale_stuff)),
    # ('check_stuff_3', FunctionTransformer(check_stuff_3)),
    ('f_transformer', f_transformer),
    # ('filter_stuff', FunctionTransformer(filter_stuff)),
    # ('check_stuff_3', FunctionTransformer(check_stuff_3))
])

models = [
    MLPClassifier(hidden_layer_sizes=(100,), solver='adam', activation='tanh')
]

for model in models:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    y = df4['event_action']
    x = df4.drop('event_action', axis=1)

    # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

    # predictions = pipeline.predict(x_test)
    # probs = pipeline.predict_proba(x_test)
    # scores_single_fit.append([type(model).__name__,' test roc score - ', roc_auc_score(y_test, probs[:,1])])
    # scores_single_fit.append([type(model).__name__,' test acc score - ', accuracy_score(y_test, predictions)])
    # print(type(model).__name__,' test roc score - ', roc_auc_score(y_test, probs[:,1]))
    # print(type(model).__name__,' test acc score - ', accuracy_score(y_test, predictions))
    # interm_scores.append((str(model), 'tst roc score - ', roc_auc_score(y_test, pipeline.predict_proba(x_test)[:,1])))
    # interm_scores.append((str(model), 'test acc score - ', accuracy_score(y_test, predictions)))

    pipe_summary = 'ad_camp V, resol V, cntry v2 V, cntry V, ct v2 V, ct V, brand v2 V, 200k 50/50'
    score = cross_val_score(pipeline, x, y, cv=4, scoring='roc_auc')
    cv_scores.append([round(score.mean(), 4), pipe_summary])
    pipeline.fit(x, y)

with open('models/pipeline_3.pkl', 'wb') as file:
    dill.dump({
        'model': pipeline,
        'metadata': {
            'name': 'sber_auto_sub_model_1',
            'author': 'well... me, i guess:)',
            'version': 0.00000000000000000001,
            'type': type(pipeline.named_steps["classifier"]).__name__,
            'roc-auc': cv_scores[-1][0]
        }
    }, file)

# with open('models/pipeline_200k_50_50_tst.pkl', 'rb') as file:
#     pipe_tst = dill.load(file)
#
print(cv_scores)
