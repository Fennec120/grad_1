import dill
import pandas as pd
import fastapi
from pydantic import BaseModel
import funcs

app = fastapi.FastAPI()

with open('models/pipeline_2.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    session_id: object
    hit_date: object
    hit_time: float
    hit_number: int
    hit_type: object
    hit_referer: object
    hit_page_path: object
    event_category: object
    event_action: int
    event_label: object
    event_value: float
    client_id: float
    visit_date: object
    visit_time: object
    visit_number: int
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_model: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object


class Prediction(BaseModel):
    client_id: int
    pred: int
    real_value: int
    type: tuple

class Test(BaseModel):
    zalupa: str

class tst_list(BaseModel):
    client: int
    pred: int
    real_value: int

@app.get('/status')
def status():
    return "i'm OK2"

@app.post('/test', response_model=tst_list)
def tst_2(test: Form):
    df = pd.DataFrame.from_dict([test.dict()])
    y = model['model'].predict(df)
    df2 = funcs.event_action(df)

    return {'client': test.client_id, 'pred': y[0], 'real_value': df2.event_action}

@app.get('/version')
def version1():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    df2 = funcs.event_action(df)

    return {
    "client_id": form.client_id,
    "pred": y[0],
    "real_value": df2.event_action, #form.event_action,
    "type": ('presented by ', model["metadata"]["type"])
    }

