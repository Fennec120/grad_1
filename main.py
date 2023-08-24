import dill
import pandas as pd
import fastapi
from pydantic import BaseModel
import funcs


app = fastapi.FastAPI()

with open('models/pipeline_2_200k_backup.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    event_action: int
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
    pred: int
    real_value: int
    type: str

@app.get('/status')
def status():
    return "i'm OK"

@app.post('/predict', response_model=Prediction)
def predict_mini(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    df2 = funcs.event_action(df)

    return {
    "pred": y[0],
    "real_value": df2.event_action,
    "type": (model["metadata"]["type"])
    }
