import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Apple Detection Function
# -----------------------------
@st.cache_data
def detect_apples(image_bytes, draw_boxes=True, min_area=500):
    """
    Detect red and green apples in an image and draw bounding boxes.
    More robust: HSV ranges, Gaussian blur, circularity check, padded boxes.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image format")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red apple mask
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Green apple mask
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks
    mask = cv2.bitwise_or(mask_red, mask_green)

    # Smooth mask to reduce noise
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    apple_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4*np.pi*area/(perimeter*perimeter) if perimeter > 0 else 0
        if circularity < 0.6:
            continue  # skip elongated/irregular shapes

        # Bounding box with padding
        x, y, w, h = cv2.boundingRect(contour)
        padding = int(0.1 * max(w,h))  # 10% padding
        if draw_boxes:
            cv2.rectangle(output_image,
                          (max(x-padding,0), max(y-padding,0)),
                          (x+w+padding, y+h+padding),
                          (0, 255, 0), 2)
        apple_count += 1

    # Encode image to bytes
    _, encoded_image = cv2.imencode('.jpg', output_image)
    return encoded_image.tobytes(), apple_count


# -----------------------------
# Streamlit Web App
# -----------------------------
st.set_page_config(page_title="Apple Detector 🍎", page_icon="🍏", layout="wide")

st.title("🍏 Apple Detection App")
st.markdown(
    """
    Upload an image and this app will detect **red and green apples**.
    You can choose to show bounding boxes around detected apples using the checkbox.
    """
)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")

# COMPONENT 1 (Input widget): Checkbox to toggle bounding boxes
show_boxes = st.sidebar.checkbox(
    "Show Bounding Boxes", value=True, help="Toggle drawing green boxes around detected apples"
)

# --- File Upload ---

# COMPONENT 2 (Input widget): File uploader for image
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # SAFER: use getvalue() to get bytes
    image_bytes = uploaded_file.getvalue()

    # Display uploaded image
    st.image(image_bytes, caption="📥 Uploaded Image", use_column_width=True)

    # COMPONENT 3 (Status element): Progress bar
    progress_bar = st.progress(0)
    progress_bar.progress(20)

    # COMPONENT 4 (Status element): Spinner
    with st.spinner("🔍 Detecting apples..."):
        time.sleep(0.3)
        progress_bar.progress(50)
        result_bytes, apple_count = detect_apples(image_bytes, draw_boxes=show_boxes)
        progress_bar.progress(80)
        result_image = Image.open(BytesIO(result_bytes))
        time.sleep(0.2)
        progress_bar.progress(100)

    # --- Display Results ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Processed Image")
        st.image(result_image, caption="Detected Apples", use_column_width=True)
    with col2:
        st.subheader("📊 Detection Results")

        # COMPONENT 5 (Data element): Metric showing number of apples
        st.metric("Number of Apples Detected", apple_count)

        if apple_count > 0:
            st.success("✅ Apples detected!")
        else:
            st.warning("No apples found.")

else:
    st.info("⬆️ Please upload an image to start detection.")

# --- Footer ---
st.markdown("---")
st.markdown("Assignment #2 built for INFT-41000: By Siddharth Santhosh")
