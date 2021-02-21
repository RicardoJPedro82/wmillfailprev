

import streamlit as st



if st.button('Get Predictions'):
    # print is visible in server output, not in the page
    print(' ')
    st.write('Predictions: 🎉')
    st.write('Further clicks are not visible but are executed')
else:
    st.write(' ')


st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Enter CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
