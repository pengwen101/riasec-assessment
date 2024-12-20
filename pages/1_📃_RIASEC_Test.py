import streamlit as st
import pandas as pd
import os

# Membaca file CSV
questions_df = pd.read_csv("./questions/holland-questions.csv")

st.title("RIASEC Assessment")

with st.form("riasec_form"):
    st.write("Jawablah pertanyaan berikut dengan skala 1-5: \n")
    st.write("**(1 = Sangat Tidak Yakin, 2 = Tidak Yakin, 3 = Netral, 4 = Yakin, 5 = Sangat Yakin)**")

    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    
    st.subheader("Questions:")
    
    for index, row in questions_df.iterrows():
        question = row['Question']
        st.session_state.answers[f"answer_{index}"] = st.slider(f"{index+1}. {question}", min_value=1, max_value=5, format="%d", help="1 = Sangat Tidak Yakin, 2 = Tidak Yakin, 3 = Netral, 4 = Yakin, 5 = Sangat Yakin")
    
    st.markdown("""
            <style>
            div.stButton > button {
                display: block;
                margin: 0 auto;
            }
            </style>
            """, unsafe_allow_html=True)
    
    submitted = st.form_submit_button("Submit")

if submitted:
    total_scores = {type_: 0 for type_ in questions_df['Type'].unique()}
    for index in range(len(questions_df)):
        type_ = questions_df.loc[index, 'Type']
        score = st.session_state.answers[f'answer_{index}']
        total_scores[type_] += score
    
    
    # Menyimpan hasil ke file CSV baru
    total_scores_df = pd.DataFrame(total_scores.items(), columns=['Type', 'Total Score'])
    print(total_scores_df)
    directory = './answers'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = './answers/riasec_assessment_answer.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    total_scores_df.to_csv(file_path, index=False)
    st.success("Jawaban anda telah berhasil disimpan!")
    
    st.session_state.top_3 = total_scores_df.sort_values(by='Total Score', ascending=False).head(3)
    # Displaying the results in a box with st.markdown and table

if 'top_3' in st.session_state:
    st.markdown(
            f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
                <h3 style="color: #4CAF50;">Hasil Tes Anda</h3>
                <h5>{st.session_state.top_3.iloc[0, 0]}</h5>
                <h5>{st.session_state.top_3.iloc[1, 0]}</h5>
                <h5>{st.session_state.top_3.iloc[2, 0]}</h5>
            </div>
            """, unsafe_allow_html=True
    )
    