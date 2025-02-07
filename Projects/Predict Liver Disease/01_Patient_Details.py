import pandas as pd
import streamlit as st
import pickle


st.set_page_config(
    page_title="Enter Patient Details...",
)


# load the standardscaler 
sc = pickle.load(open('artifacts/scaler.sav', 'rb')) 

# load the model 
knn = pickle.load(open('artifacts/knn.sav', 'rb')) 

def converNumericalCategoryToStandard(value):
      if value == 0:
        diagnosis = 'is diagnosed with cirrhosis. Follow up with doctor immediately.'
        img = "img/cirrhosis.png"
      elif value == 1:
        diagnosis =  'is diagnosed with fibrosis. Follow up with doctor immediately.'
        img = "img/fibrosis.png"
      elif value == 2: 
        diagnosis = 'is diagnosed with hepatitis. Follow up with doctor immediately.'
        img = "img/hepatitis.png"
      elif value == 3:
        diagnosis = 'has no disease.'
        img = "img/no_disease.webp"
      elif value == 4:
        diagnosis = 'has suspect_disease. Need further testing.'
        img = "img/disease.png"
      return diagnosis, img



st.markdown("<h2 style='text-align: center; color: white;'>Fill in your patient data here for diagnosis</h2>", unsafe_allow_html=True)

with st.form("patient_form"):
    st.markdown("<h6 style='text-align: color: white;'>Enter Patient Details:</h6>", unsafe_allow_html=True)
    name = st.text_input("Patient Name", placeholder="John Doe")

    col1, col2 = st.columns(2)

    age = col1.number_input("Age", value = 47, help = "The average age for onset liver disease is 47 years.")
    sex = col2.selectbox('Gender', ('Male', 'Female'), help = "Men are more likely to develop liver disease than women.")
    albumin = col1.number_input("Albumin", value = 41.62, help = "Protein synthesized by the liver, contributes to immune function and acts as a binding agent for certain molecules (g/L)")
    alkaline_phosphatase = col2.number_input("Alkaline Phosphatase", value = 68.22, help = "An enzyme, plays an integral role in metabolism within the liver and development within the skeleton (u/L)")
    alanine_aminotransferase = col1.number_input("Alanine Aminotransferase", value = 28.44, help = "An important enzyme in the intermediary metabolism of glucose and protein (u/L)")
    aspartate_aminotransferase = col2.number_input("Aspartate Aminotransferase" , value = 34.78, help = "An enzyme in amino acid metabolism found in liver, heart, skeletal muscle, etc. (u/L)")
    bilirubin = col1.number_input("Bilirubin" , value = 11.39, help = "Yellow pigment produced during the breakdown of red blood cells (μmol/L)")
    cholinesterase = col2.number_input("Cholinesterase", value = 8.19, help = "An enzyme, involved in the breakdown of acetylcholine, a neurotransmitter (100u/L)")
    cholesterol = col1.number_input("Cholesterol", value = 5.36, help = "A fatty substance produced by the liver and obtained from dietary sources (mmol/L)")
    creatinina = col2.number_input("Creatinina", value = 81.28, help = "A waste product produced by muscles during normal metabolism (μmol/L)")
    gamma_glutamyl_transferase = col1.number_input("Gamma Glutamyl Transferase" , value = 39.53, help = "Liver enzyme, responsible for absorption of amino acids from the glomerular filtrate and from the intestinal lumen (u/L)")
    protein = col2.number_input("Protein", value = 72.04, help = "An essential macronutrient involved in various physiological processes, including tissue repair and immune function (g/L)")

    submitted = st.form_submit_button("Submit")

if submitted:
        st.markdown("<h6 style='text-align: color: white;'>Diagnosis:</h6>", unsafe_allow_html=True)
        # initialize data of lists.
        data = {'age': age, 'albumin': albumin, 'alkaline_phosphatase': alkaline_phosphatase, 'alanine_aminotransferase':alanine_aminotransferase, 
                'aspartate_aminotransferase':aspartate_aminotransferase, 'bilirubin': bilirubin, 'cholinesterase': cholinesterase, 'cholesterol': cholesterol,
                'creatinina': creatinina, 'gamma_glutamyl_transferase': gamma_glutamyl_transferase, 'protein': protein
                }
        df = pd.DataFrame(data, index=[0])
        df = sc.transform(df)
        if sex == "Male":
              df['sex_m'] = 1
              df['sex_f'] = 0
        else:
              df['sex_m'] = 0
              df['sex_f'] = 1
        diagnosis , img = converNumericalCategoryToStandard(knn.predict(df))
        st.image(img, width=200)
        st.write("Patient ", name , diagnosis)

