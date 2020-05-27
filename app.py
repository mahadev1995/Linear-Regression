from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np 



app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():
    data = pd.read_csv("kc_house_data.csv")
    columns = ["bedrooms", "bathrooms", "sqft_lot", "floors", "yr_built"]
    X = data[columns]
    y = data["price"]
    from sklearn.linear_model import LinearRegression
    rgr = LinearRegression()
    rgr.fit(X, y)
    

    if request.method == 'POST':
        bedrooms = request.form['bedroom']
        bathrooms = request.form['bathroom']
        area = request.form['landarea']
        floors = request.form['floor']
        year = request.form['year']

        data = pd.DataFrame([[bedrooms, bathrooms, area, floors, year]])
        my_prediction = rgr.predict(data)
    return render_template('results.html', prediction = int(round(my_prediction[0])),bed = bedrooms, bath = bathrooms, floor = floors, yr = year,land = area)

if __name__ == '__main__':
    app.run(debug = True)

