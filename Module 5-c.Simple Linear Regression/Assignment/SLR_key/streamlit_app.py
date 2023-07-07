import pandas as pd
import streamlit as st 
# import numpy as np

from sqlalchemy import create_engine
from urllib.parse import quote
import joblib #, pickle

impute = joblib.load('imp')
winsor = joblib.load('winsor')
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")


def predict_AT(data,user,pw,db):
    data.rename(columns = {'Sorting Time':'Sorting_Time'}, inplace = True)
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
                    
    clean1 = pd.DataFrame(impute.transform(data), columns = ['Sorting_Time'])   
    clean1[['Sorting_Time']] = pd.DataFrame(winsor.transform(clean1[['Sorting_Time']]))    
    prediction = pd.DataFrame(model.predict(clean1), columns = ['Delivery_Time'])
    
    final = pd.concat([prediction, data], axis = 1)
    final.to_sql('mpg_predictons', con = engine, if_exists = 'replace', chunksize = 1000, index= False)

    return final



def main():
    st.title("AT prediction")
    st.sidebar.title("Fuel Efficiency prediction")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">AT prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html = True)
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here",  type="password")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_AT(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()


