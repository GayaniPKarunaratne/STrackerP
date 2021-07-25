from flask import Flask, request
import pandas as pd
from _collections import OrderedDict
import joblib

app = Flask(__name__)

@app.route("/api/stracker")
def get() :
    HandWashing = float(request.args["HandWashing"])
    MaskWearing = float(request.args["MaskWearing"])
    ScocialDistence = float(request.args["ScocialDistence"])
    BehaveInPublicPlaces = float(request.args["BehaveInPublicPlaces"])
    Symptoms = float(request.args["Symptoms"])

    folder_name = "model/"
    file_path = (folder_name + "stracker.joblib")
    file = open(file_path, "rb")
    model_load = joblib.load(file)

    request_data = OrderedDict([("HandWashing",HandWashing),("MaskWearing",MaskWearing),("ScocialDistence",ScocialDistence),("BehaveInPublicPlaces",BehaveInPublicPlaces),("Symptoms",Symptoms)])
    reshaped_data = pd.Series(request_data).values.reshape(1,-1)
    prediction = model_load.predict(reshaped_data)
    return str(prediction)

if __name__ == "__main__":
    app.run()
