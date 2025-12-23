"""API do przewidywania wartości zawodników"""

from fastapi import FastAPI
import onnxruntime as rt
import numpy as np
from schemas import FantasyAcquisitiuonFeatures, PredictOutput

api_description = """
API przechowuje zakres kosztów pozyskania zawodnika.
Punkty końcowe można podzielić na poniższe kategorie:

## Analityka
Uzyskaj informacje o stanie API

## Prognozy
Uzyskaj prognozy kosztów pozyskania zawodnika"""

#Wczytanie modelu ONNX
sess_10 = rt.InferenceSession("acquisition_model_10.onnx",
                              providers=["CPUExecutionProvider"])
sess_50 = rt.InferenceSession("acquisition_model_50.onnx",
                              providers=["CPUExecutionProvider"])
sess_90 = rt.InferenceSession("acquisition_model_90.onnx",
                              providers=["CPUExecutionProvider"])
input_name_10 = sess_10.get_inputs()[0].name
label_name_10 = sess_10.get_outputs()[0].name
input_name_50 = sess_50.get_inputs()[0].name
label_name_50 = sess_50.get_outputs()[0].name
input_name_90 = sess_90.get_inputs()[0].name
label_name_90 = sess_90.get_outputs()[0].name

app = FastAPI(
    description=api_description,
    title="API do zarządzania drużyną Fantasy",
    version="0.1"
)

@app.get(
    "/",
    summary="Sprawdzenie, czy API działa",
    description="Użyj tego punktu aby sprawdzić czy API działa",
    response_description="Obiekt JSON zawierający komunikat. " \
    "Jeśli API działa, będzie to komunikat o sukcesie",
    operation_id="v0_health_check",
    tags=["analityka"]
)
def root():
    return {"message": "Test stanu API zkończony sukcesem"}

# Utworzenie predykcji
@app.post("/predict",
          response_model=PredictOutput,
          summary="Przewidywanie kosztu pozyskania zawodnika",
          description = """Użyj tego punktu końcowego, aby przewidzieć
          zakres kosztów pozyskania zawodnika""",
          response_description="""Rekord JSON zaiwerający trzy przewidywane kwoty.\
            Razem tworzą możliwy zakres kosztu zakupu zawodnika""",
            operation_id="v0_predict",
            tags=["predykcja"])
def predict(features: FantasyAcquisitiuonFeatures):
    # konwersja modelu na tablicę Numpy
    input_data = np.array([[features.waiver_value_tier,
                            features.fantasy_regular_season_weeks_remaining,
                            features.league_budget_pct_remaining]],
                            dtype = np.int64)
    pred_onx_10 = sess_10.run([label_name_10], {input_name_10: input_data})[0]
    pred_onx_50 = sess_50.run([label_name_50], {input_name_50: input_data})[0]
    pred_onx_90 = sess_90.run([label_name_90], {input_name_90: input_data})[0]

    # Zwrócenie predykcji w postaci obiektu Pydantic
    return PredictOutput(winning_bid_10th_percentile=round(
        float(pred_onx_10[0]),2),
        winning_bid_50th_percentile=round(
            float(pred_onx_50[0]),2),
        winning_bid_90th_percentile=round(
            float(pred_onx_90[0]),2
        )
    )