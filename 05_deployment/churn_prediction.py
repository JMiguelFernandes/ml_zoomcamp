import pickle
from flask import Flask, request, jsonify

with open("dv.bin", "rb") as file_in:
    dv = pickle.load(file_in)
    
with open("model2.bin", "rb") as file_in:
    model1 = pickle.load(file_in)
    
app = Flask("/churn_prediction")

@app.route("/churn_prediction", methods=["POST"])
def churn_prediction():
    customer = request.get_json()
    result = {
        "churn_probability" : model1.predict_proba(dv.transform(customer))[0,1]
    }
    return(jsonify(result))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)