import cv2
import numpy as np
import streamlit as st

st.write("""
OpenCV Useful methods
""")

uploaded_file = st.file_uploader("Choose a image file", type=['jpg', 'jpeg', 'png', 'jfif'])

# Useful methods in opencv
st.sidebar.header("Play with the methods to apply on your image")

# translate an image to new x, y positions
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)

# rotate an image by an angle
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv2.warpAffine(img, rotMat, dimensions)


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.write("""### Original image:""")
    st.image(opencv_image, channels="BGR")


    # Color conversions
    st.sidebar.subheader("Color conversions")
    selected_color_conv = st.sidebar.selectbox("Select", ["GRAY", "HSV"])
    if selected_color_conv == "GRAY":
        colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        st.write("""### Grayscale image:""")
        st.code("""
        colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        """, language='python')
        st.image(colored_img)
    elif selected_color_conv == "HSV":
        colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        st.write("""### Hue Saturated image:""")
        st.code("""
        colored_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        """, language='python')
        st.image(colored_img)

    # Gaussian blur
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    st.sidebar.subheader("Gaussian Blur")
    selected_kernel_size = st.sidebar.slider('Kernel size', 1, 49, step=2)
    kernel_size = (selected_kernel_size, selected_kernel_size)
    blurred_img = cv2.GaussianBlur(opencv_image_rgb, kernel_size, cv2.BORDER_DEFAULT)
    st.write("""### Gaussian Blur image:""")
    st.write(f"Kernel size: {(kernel_size)}")
    st.code("""
    blurred_img = cv2.GaussianBlur(opencv_image_rgb, kernel_size, cv2.BORDER_DEFAULT)
    """, language='python')
    st.image(blurred_img)

    # Canny edges
    st.sidebar.subheader("Canny edges")
    threshold1 = st.sidebar.slider("Threshold 1", 1, 500, step=1, value=300)
    threshold2 = st.sidebar.slider("Threshold 2", 1, 500, step=1, value=200)
    selected_canny_img = st.sidebar.selectbox("Apply on:", ["Original image", "Blurred image"])
    if selected_canny_img.startswith("Original"):
        canny_img = cv2.Canny(opencv_image_rgb, threshold1, threshold2)
    else:
        canny_img = cv2.Canny(blurred_img, threshold1, threshold2)
    st.write("""### Canny edges detector:""")
    st.code("""
    canny_img = cv2.Canny(opencv_image_rgb, threshold1, threshold2)
    """, language='python')
    st.image(canny_img)

    # dilated image
    st.sidebar.subheader("Dilated image")
    dilated_kernel = st.sidebar.slider('Dilated Kernel size', 1, 49, step=2)
    dilated_iterations = st.sidebar.slider('Dilated Iterations:', 1, 10, step=1)
    dilated_img = cv2.dilate(canny_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    st.write("""### Dilated image:""")
    st.code("""
    dilated_img = cv2.dilate(canny_img, (dilated_kernel, dilated_kernel), iterations=dilated_iterations)
    """, language='python')
    st.image(dilated_img)

    # Erroded image
    st.sidebar.subheader("Erroded image")
    erroded_kernel = st.sidebar.slider('Erroded Kernel size', 1, 49, step=2)
    erroded_iterations = st.sidebar.slider('Erroded Iterations:', 1, 10, step=1)
    erroded_img = cv2.erode(dilated_img, (erroded_kernel, erroded_kernel), iterations=erroded_iterations)
    st.write("""### Erroded image:""")
    st.code("""
    erroded_img = cv2.dilate(canny_img, (erroded_kernel, erroded_kernel), iterations=erroded_iterations)
    """, language='python')
    st.image(erroded_img)

    # Resize image
    st.sidebar.subheader("Resize image")
    resize_width = st.sidebar.number_input("Width", 10, 2000, step=1, value=100)
    resize_height = st.sidebar.number_input("Height", 10, 2000, step=1, value=100)
    resized_img = cv2.resize(opencv_image_rgb, (resize_width, resize_height))
    st.write("""### Resize Image""")
    st.code("""
    resized_img = cv2.resize(opencv_image_rgb, (resize_width, resize_height))
    """, language='python')
    st.write(f"Taget shape: {(resize_width, resize_height)}")
    st.image(resized_img)

    # Translate image
    st.sidebar.subheader("Translate image")
    translateX = st.sidebar.number_input("translateX", -1000, 1000, value=-90)
    translateY = st.sidebar.number_input("translateY", -1000, 1000, value=250)
    translated_img = translate(opencv_image_rgb, translateX, translateY)
    st.write("""### Translate Image""")
    st.write(f"translateX: {translateX}")
    st.write(f"translateY: {translateY}")
    st.code("""
    translated_img = translate(opencv_image_rgb, translateX, translateY)

    def translate(img, x, y):
        transMat = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv2.warpAffine(img, transMat, dimensions)
    """, language='python')
    st.image(translated_img)

    # Rotate image
    st.sidebar.subheader("Rotate image")
    rotate_angle = st.sidebar.number_input("Rotate by angle:", -1000, 1000, value=45)
    rotate_pointX = st.sidebar.number_input("Rotate point x_coord", -1000, 1000, step=1, value=500)
    rotate_pointY = st.sidebar.number_input("Rotate point y_coord", -1000, 1000, step=1, value=500)
    rotated_img = rotate(opencv_image_rgb, rotate_angle, (rotate_pointX, rotate_pointY))
    st.write("""### Rotate Image""")
    st.write(f"Rotation Point: {(rotate_pointX, rotate_pointY)}")
    st.write(f"Rotation angle: {rotate_angle}")
    st.code("""
    rotated_img = rotate(opencv_image_rgb, rotate_angle, (rotate_pointX, rotate_pointY))

    def rotate(img, angle, rotPoint=None):
        (height, width) = img.shape[:2]

        if rotPoint is None:
            rotPoint = (width // 2, height // 2)
        
        rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
        dimensions = (width, height)

        return cv2.warpAffine(img, rotMat, dimensions)
    """, language='python')
    st.image(rotated_img)

    # Flip image
    st.sidebar.subheader("Flip Image")
    flip_choice = st.sidebar.selectbox("Flip choice:", ["Flip Vertically", "Flip Horizontally", "Flip both"])
    if flip_choice == "Flip Vertically":
        flipCode = 0
    elif flip_choice == "Flip Horizontally":
        flipCode = 1
    else:
        flipCode = -1
    flipped_img = cv2.flip(opencv_image_rgb, flipCode)
    st.write("""### Flip image """)
    st.write(flip_choice)
    st.code(f"""
    flipped_img = cv2.flip(opencv_image_rgb, flipCode={flipCode})
    """, language='python')
    st.image(flipped_img)

    st.sidebar.subheader("And much more...")

