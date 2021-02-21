
import joblib
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import base64
from PIL import Image


def main_teste():

    st.markdown("## ML Project")#, FONT_SIZE_CSS, unsafe_allow_html=True)
    st.markdown("## **Blown in the Wind**")
    st.markdown("Evaluate if the following turbine components are likely to fail in the next **60 days**: ")
    st.markdown("""

    - *Generator*
    - *Hydraulic*
    - *Generator Bearing*
    - *Transformer*
    - *Gearbox*
""")

    #font_style=f"""<font color=‘red’>THIS TEXT WILL BE RED</font>"""
    #st.write(font_style, unsafe_allow_html=True)

    st.write(' ')
    st.write(' ')

    st.set_option('deprecation.showfileUploaderEncoding', False)

    uploaded_file = st.file_uploader("Enter CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep=";")
        st.write(data)


    @st.cache
    def load_image(path):
        with open(path, 'rb') as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        return encoded

    def background_image_style(path):
        encoded = load_image(path)
        style = f'''
        <style>
        body {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        '''
        return style

    image_path = 'images/Wind-Turbines03-1170x820.jpg'


    st.write(background_image_style(image_path), unsafe_allow_html=True)

    if st.button('Get Predictions'):
        # print is visible in server output, not in the page
        print(' ')
        st.write('Predictions:')
        st.write(' ')

        @st.cache
        def get_bar_chart_data():
            print('get_bar_chart_data called')
            return pd.DataFrame(
                np.random.randn(6, 5),
                columns=["T01", "T06", "T07", "T09", "T11"]
            )

        chart_data = get_bar_chart_data()

        st.bar_chart(chart_data)

    else:
        st.write(' ')





# print(colored(proc.sf_query, "blue"))
# proc.test_execute()
if __name__ == "__main__":
    #df = read_data()
    main_teste()
