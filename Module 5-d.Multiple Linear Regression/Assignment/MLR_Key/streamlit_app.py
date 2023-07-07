import pandas as pd
import streamlit as st 
# import numpy as np
from urllib.parse import quote 
from sqlalchemy import create_engine
import pickle, joblib
from statsmodels.tools.tools import add_constant


model1 = pickle.load(open('profit.pkl','rb'))
imp_enc_scale = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')


def predict_profit(data, user, pw, db):

    engine = create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))

    clean = pd.DataFrame(imp_enc_scale.transform(data), columns = imp_enc_scale.get_feature_names_out())
    
    clean[['num__R&D Spend', 'num__Administration', 'num__Marketing Spend']] = winsor.transform(clean[['num__R&D Spend', 'num__Administration', 'num__Marketing Spend']])
    
    clean = add_constant(clean)
    
    prediction = pd.DataFrame(model1.predict(clean), columns = ['MPG_pred'])
    
    prediction.to_sql('profit_predicton', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return prediction



def main():

    st.title("Profit prediction of Startup")
    st.sidebar.title("Profit prediction of Startup")

   
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Profit prediction of startup</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

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
    <p style="color:white;text-align:center;">Add DataBase Credientials</p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_profit(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()

