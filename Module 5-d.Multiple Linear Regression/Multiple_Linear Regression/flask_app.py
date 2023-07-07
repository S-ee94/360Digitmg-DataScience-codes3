import pandas as pd
from flask import Flask, render_template, request
from sqlalchemy import create_engine
import joblib, pickle

model1 = pickle.load(open('mpg.pkl', 'rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "Seemscrazy1994#", # passwrd
                               db = "cars_db")) #database


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_csv(f)
        clean = pd.DataFrame(impute.transform(data),columns=data.select_dtypes(exclude = ['object']).columns)
        clean1 = winsor.transform(clean)
        clean2 = pd.DataFrame(minmax.transform(clean1))
        clean3 = pd.DataFrame(encoding.transform(data).todense())
        clean_data = pd.concat([clean2, clean3], axis = 1, ignore_index = True)
        clean_data1 = clean_data.drop(3, axis = 1)
        prediction = pd.DataFrame(model1.predict(clean_data1), columns = ['MPG'])
        
        final = pd.concat([prediction,data], axis = 1)
        final.to_sql('mpg_predictons', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        return render_template("data.html", Z = "Your results are here" , Y = final.to_html(justify='center').replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bordercolor="#000000" bgcolor="#FFCC66">'))

if __name__ == '__main__':

    app.run(debug = True)
