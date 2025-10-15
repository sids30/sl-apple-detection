# Apple Detection Streamlit App 🍎

Deployed Streamlit App Link: https://apple-detect.streamlit.app/

This interactive web application detects **red and green apples** in uploaded images using **OpenCV and color segmentation**.  
It draws **bounding boxes** around the apples and displays the **total number of apples detected**.

---

## 🧠 Model Description

- The app uses a **color-based detection model**:
  - Converts images to **HSV color space**.
  - Creates masks for **red and green apples** using wide HSV ranges to handle different lighting conditions.
  - Applies **Gaussian blur** and **morphological operations** to remove noise.
  - Detects contours and filters by **area** and **circularity** to identify roughly circular objects.
  - Draws **bounding boxes** around detected apples.
- This approach is fast and works for images with multiple apples.

---

## 🚀 Running the App Locally

1. **Clone the repository**:

```bash
https://github.com/sids30/st_apple_detetction.git
cd st_apple_detection
```

2. Create a Virtual Environment
   
   1) python -m venv venv
   2) source venv/bin/activate (MacOS) OR venv\Scripts\activate (Windows)


3. Install Dependencies

   - pip install -r requirements.txt
  

4. Run Streamlit App

   - streamlit run app.py


