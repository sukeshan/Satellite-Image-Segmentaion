import streamlit as st
import cv2
import numpy as np
from prediction import predict
import time
import matplotlib.pyplot as plt
from watershed import water_shed
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import pandas as pd

st.set_page_config(
    page_title=" ടëɕ",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)
def df(props):
    gb = GridOptionsBuilder.from_dataframe(props)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    a = gb.configure_selection('single', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(props, gridOptions=gridOptions, data_return_mode='AS_INPUT',update_mode='MODEL_CHANGED', fit_columns_on_grid_load=False,
        theme='streamlit', #Add theme color to the table
        enable_enterprise_modules=True, height=350,  width='1050%', reload_data=True)
    data = grid_response['data']
    selected_rec = grid_response['selected_rows'] 



def pie(sizes):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Buildings','Others'
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes/5000, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig1

col1  ,col2 = st.columns([4, 5])

st.markdown("<h1 style='text-align: center; color: Grey;font-size: 40px;'> Satellite Image Segmentation </h1>", unsafe_allow_html=True)
#c2.image(cv2.cvtColor(cv2.imread('logo.jpeg'),cv2.COLOR_BGR2GRAY) ,width = 350)
img_file = st.file_uploader("Choose a image file")

if img_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
c1 ,c2 =  st.columns([3.5, 5])

if c2.button('Predict'):
    st.write('Your file has been sucessfully uploaded')

    st.markdown("<h1 style='text-align: center; color: Grey;font-size: 20px;'> Due to Hardware Constrain ,It will process Slowly 😥...</h1>", unsafe_allow_html=True)
    
    mapped_img ,area ,predicted = predict(img) # Unet Prediction -------------------
    st.success('Done')
    col1  ,col2 ,col3 = st.columns([1, 1 ,1])
    
    col1.image(img_file,channels='BGR',width = 300 ,caption = 'Given Image ')
    col2.image(mapped_img,channels='BGR',width = 300 ,caption = 'Predicted Image')
    col3.pyplot(pie(area))
    c4 ,c5 = st.columns([2,3])
    overly_img  ,insta ,props = water_shed(predicted ,img) # calling watershed-----------------
    
    #c4.image(overly_img ,channels='BGR',width = 300 ,caption = 'Detected Buildings')
    #c5.image(insta,channels='BGR',width = 300 ,caption = 'Insta Segmented image')
    
    df(props)
    c1 ,c2 =  st.columns([3.5, 5])

    if c2.button('Want to save'):
        cv2.imwrite('predicted.jpeg',predicted) 
        pd.to_csv('Report.csv',props)

    #st.image(predicted,channels='BGR',width = 300)