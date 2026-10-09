import streamlit as st
import pandas as pd
import joblib

model=joblib.load('logistic_model.pkl')
scaler=joblib.load('scaler.pkl')
required_col=joblib.load('columns.pkl')

st.header('Heart Dieases Prediction Model')

st.markdown('please Provide folloving details for predict Heart Dieases')

#! Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
#! 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'

age=st.slider("Age", 10, 100, 20)
sex=st.selectbox('Gender', ['M', 'F'])
chestpaintype=st.selectbox('ChestPainType', ['ATA', 'NAP', 'ASY', 'TA'])
restingbp=st.number_input('RestingBP', 0, 200, 125)
cholesterol=st.number_input('Cholesterol', 0)
fastingbs=st.selectbox('FastingBS', [0, 1])
restingECG=st.selectbox('RestingECG',['Normal', 'ST', 'LVH'])
maxhr=st.number_input('MaxHR', 0, 300, 121)
exerciseangina=st.selectbox('ExerciseAngina', ['N', 'Y'])
oldpeak=st.slider('Oldpeak', -10, 10)
st_slop=st.selectbox('ST_Slope',['Up', 'Flat', 'Down'])

if st.button('predict'):
    input_data={
        'Age':age,
        'Sex_'+sex:1,
        'ChestPainType_'+chestpaintype:1,
        'RestingBP':restingbp,
        'Cholesterol':cholesterol,
        'FastingBS':fastingbs,
        'RestingECG_'+restingECG:1,
        'MaxHR':maxhr,
        'ExerciseAngina_'+exerciseangina:1,
        'Oldpeak':oldpeak,
        'ST_Slope_'+st_slop:1
    }
    df=pd.DataFrame([input_data])
    for col in required_col:
        if col not in df.columns:
            df[col]=0

    df=df[required_col]

    X_scaled=scaler.transform(df)

    predict=model.predict(X_scaled)
    # st.markdown(predict[0])
    if (predict[0] == 0):
        st.success('Low Chance of Heart Dieases')
    else:
        st.warning('High Chance Of Heart Dieases')    


        











































































































