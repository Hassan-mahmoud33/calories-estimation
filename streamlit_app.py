import numpy  as np 
import cv2
from PIL import Image
import streamlit as st
from real_time import model , CAL_PER_100G , GRAMS_PER_PIXEL

#------------------------------------------


uploaded_file = st.file_uploader('Insert An Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert('RGB')
    img = cv2.resize(np.array(img), (900, 620))
    st.image(img)

    if st.button('Detect'):

        results = model(img , conf=0.25)
        r = results[0]
        annotated = r.plot()

        st.image(annotated)

        total_calories = 0
        if r.masks is not None:

            for i , poly in enumerate(r.masks.xy):
                if len(poly) < 3: # must at least 3 points to form a polygon
                    continue

                cls_id = int(r.boxes.cls[i])
                cls_name = model.names[cls_id]

                area_px = cv2.contourArea(poly.astype(np.int32))

                grams_divided100 = (area_px * GRAMS_PER_PIXEL) / 100
                calories = int( grams_divided100 * CAL_PER_100G[cls_name] )

                total_calories += calories


        st.write(f"Total Calories: {total_calories}")
