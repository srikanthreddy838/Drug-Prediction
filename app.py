from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    def out(prediction):
        drug_names=[]
        if prediction==0:
            drug_names.append('DrugA')
        elif prediction[0]==1:
            drug_names.append('DrugX')
        elif prediction[0]==2:
            drug_names.append('DrugY')
        elif prediction[0]==3:
            drug_names.append('DrugB')
        else:
            drug_names.append('DrugC')
        return drug_names
    drugname=out(prediction)
    return render_template('predict.html', prediction_text='{}'.format(drugname))

    
if __name__ == "__main__":
    app.run(debug=True)