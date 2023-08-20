import streamlit as st
import numpy as np
import cv2
import streamlit.components.v1 as components


st.set_page_config(page_title="Image Cartoonifier", layout="wide")
st.title("Cartoonify Image Using Deep Learning")
st.text("")
st.header("Welcome To Our Portal")

# Components Bootstrap

components.html(
    """
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
<div class="card" style="width: 100% , height:80%;">
  <img class="card-img-top" src="https://monomousumi.com/wp-content/uploads/67443208_2456472761254037_664854829977305088_o.jpg" alt="Card image cap">
  
""",
    height=400,
)

upload = st.file_uploader("Upload your image", type=["JPG", "JPEG", "PNG"])
if upload is not None:
    with open("image.jpg", "wb") as f:
        f.write(upload.getbuffer())

    st.image("image.jpg", caption='Original Image', width=400)
    img = cv2.imread("image.jpg")

   # edge mask generation
    line_size = 7
    blur_value = 7

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray_img, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

    st.image(edges, caption='Edges Image', width=400)

   # Color quantization with KMeans clustering
    from sklearn.cluster import KMeans

    k = 7
    data = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    img_reduced = kmeans.cluster_centers_[kmeans.labels_]
    img_reduced = img_reduced.reshape(img.shape)

    img_reduced = img_reduced.astype(np.uint8)

    st.image(img_reduced, caption='Reduced Image', width=400)

   # Bilateral Filter
    blurred = cv2.bilateralFilter(
        img_reduced, d=5, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    cartoon_image = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)

    st.image(cartoon_image, caption='Processed Image', width=400)
    cv2.imwrite('cartoon_image.jpg', cv2.cvtColor(
        cartoon_image, cv2.COLOR_RGB2BGR))
    with open("cartoon_image.jpg", "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="cartoon.jpg",
            mime="image/jpg")
