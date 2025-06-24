import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imutils
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="UI Comparator", layout="wide")

def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        raise Exception("Not enough matches found to align images.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    height, width = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, H, (width, height))
    return aligned_img1

def compare_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    result_img = img2.copy()
    for c in cnts:
        if cv2.contourArea(c) > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return result_img, score

def to_image_bytes(img):
    is_success, buffer = cv2.imencode(".png", img)
    return buffer.tobytes() if is_success else None

def main():
    st.title("🧪 UI Screenshot Comparator")
    st.markdown("Compare Figma design vs. actual app screenshot using ORB + SSIM")

    col1, col2 = st.columns(2)
    with col1:
        figma_file = st.file_uploader("📁 Upload Figma Screenshot", type=["png", "jpg", "jpeg"])
    with col2:
        app_file = st.file_uploader("📁 Upload Actual App Screenshot", type=["png", "jpg", "jpeg"])

    if figma_file and app_file:
        figma_image = np.array(Image.open(figma_file).convert("RGB"))
        app_image = np.array(Image.open(app_file).convert("RGB"))

        fixed_size = (1080, 1920)
        figma_image = cv2.resize(figma_image, fixed_size)
        app_image = cv2.resize(app_image, fixed_size)

        try:
            aligned_app = align_images(app_image, figma_image)
            result_img, score = compare_images(aligned_app, figma_image)

            st.success(f"✅ SSIM Score: {score:.4f}")
            st.image([figma_image, aligned_app, result_img], caption=["Figma", "Aligned App", "Comparison"], width=300)

            st.download_button("📥 Download Aligned App", data=to_image_bytes(aligned_app),
                               file_name="aligned_app.png", mime="image/png")
            st.download_button("📥 Download Comparison Image", data=to_image_bytes(result_img),
                               file_name="comparison_result.png", mime="image/png")

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
