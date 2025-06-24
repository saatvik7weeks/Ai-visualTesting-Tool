import streamlit as st
import os
import uuid
import pandas as pd
from io import BytesIO
from compare import compare_images, ask_openai_cosmetic_diff, encode_image, ask_ui_suggestion

UPLOAD_FOLDER = "backend/uploads"
DIFF_FOLDER = "backend/diffs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DIFF_FOLDER, exist_ok=True)

# UI Config
st.set_page_config(page_title="UI Screenshot Comparator", layout="centered")
st.title("🖼️ UI Testing Analysis")

# Theme Toggle
theme = st.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"], horizontal=True)
bg_color = "#ffffff" if "Light" in theme else "#0e1117"
text_color = "#000000" if "Light" in theme else "#ffffff"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("Upload your **Figma design** and the **App screenshot**, and this tool will detect **cosmetic UI bugs**.")

figma_file = st.file_uploader("📌 Upload Figma Screenshot", type=["png", "jpg", "jpeg"], key="figma")
app_file = st.file_uploader("📱 Upload App Screenshot", type=["png", "jpg", "jpeg"], key="app")

# Dropdown for bug filtering
bug_type_filter = st.selectbox(
    "🔍 Select Bug Type to Display",
    ["All", "Missing UI Elements", "Text Errors", "Color Mismatch", "Other Cosmetic Differences"],
    index=0
)

def generate_bug_excel(bug_report: str) -> BytesIO:
    rows = []
    for line in bug_report.split("\n"):
        line = line.strip("-•* ").strip()
        if not line:
            continue
        if "missing" in line.lower():
            expected = "Element should be visible as per Figma"
            actual = "Element is missing in App screenshot"
        elif "spelling" in line.lower() or "text" in line.lower():
            expected = "Correct spelling and content"
            actual = "Text mismatch or spelling error"
        elif "color" in line.lower():
            expected = "Color should match Figma design"
            actual = "Color differs in App"
        elif "font" in line.lower():
            expected = "Font size/style should match design"
            actual = "Font style or size differs"
        else:
            expected = "Should match Figma"
            actual = "Does not match Figma"

        rows.append({
            "Bug Summary": line,
            "Expected Result": expected,
            "Actual Result": actual
        })

    df = pd.DataFrame(rows)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="BugReport")
    output.seek(0)
    return output

def format_bug_section(section_title, color, bug_report):
    lines = []
    capture = False
    for line in bug_report.split("\n"):
        if line.strip().startswith("**") and section_title.lower() not in line.lower():
            capture = False
        if section_title.lower() in line.lower():
            capture = True
            continue
        if capture and line.strip():
            cleaned = line.strip("-•* ").strip()
            if cleaned:
                lines.append(f"<li style='color:{color}; margin-bottom: 4px;'>{cleaned}</li>")
    if lines:
        return f"<h4 style='color:{color}'>{section_title}</h4><ul>{''.join(lines)}</ul>"
    return ""

if figma_file and app_file and st.button("🧪 Compare"):
    with st.spinner("Analyzing and comparing screenshots..."):
        uid = str(uuid.uuid4())
        figma_path = os.path.join(UPLOAD_FOLDER, f"{uid}_figma.png")
        app_path = os.path.join(UPLOAD_FOLDER, f"{uid}_app.png")

        with open(figma_path, "wb") as f:
            f.write(figma_file.read())
        with open(app_path, "wb") as f:
            f.write(app_file.read())

        diff_img_path, score = compare_images(figma_path, app_path)
        bug_report = ask_openai_cosmetic_diff(figma_path, app_path)

        st.markdown("### 🔍 Visual Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(figma_path, caption="🎨 Figma Design", use_column_width=True)
        with col2:
            st.image(app_path, caption="📱 App Screenshot", use_column_width=True)
        with col3:
            st.image(diff_img_path, caption="🔍 Visual Diff", use_column_width=True)

        st.markdown(f"**🔢 Similarity Score:** `{round(score, 4)}`")
        st.markdown("### 🐞 Detected Cosmetic Bugs:")

        styled_bug_report = ""

        if bug_type_filter == "All" or bug_type_filter == "Missing UI Elements":
            styled_bug_report += format_bug_section("Missing UI Elements", "#FF4B4B", bug_report)

        if bug_type_filter == "All" or bug_type_filter == "Text Errors":
            styled_bug_report += format_bug_section("Text Errors", "#3A9BDC", bug_report)

        if bug_type_filter == "All" or bug_type_filter == "Color Mismatch":
            styled_bug_report += format_bug_section("Color", "#E68600", bug_report)

        if bug_type_filter == "All" or bug_type_filter == "Other Cosmetic Differences":
            styled_bug_report += format_bug_section("Other Cosmetic Differences", "#8B008B", bug_report)

        if styled_bug_report:
            st.markdown(styled_bug_report, unsafe_allow_html=True)
        else:
            st.info("No bugs found in the selected category.")

        bug_excel = generate_bug_excel(bug_report)
        st.download_button(
            label="⬇️ Download Bug Report (Excel)",
            data=bug_excel,
            file_name="UI_Bug_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------- Suggestions Section ----------
st.markdown("---")
st.subheader("🧠 UI Suggestions (GPT-4o)")

suggest_file = st.file_uploader("💡 Upload App Screenshot for Suggestions", type=["png", "jpg", "jpeg"], key="suggestion")

if suggest_file and st.button("💬 Get Suggestions"):
    with st.spinner("Analyzing your UI for suggestions..."):
        uid = str(uuid.uuid4())
        suggest_path = os.path.join(UPLOAD_FOLDER, f"{uid}_suggest.png")

        with open(suggest_path, "wb") as f:
            f.write(suggest_file.read())

        suggestions = ask_ui_suggestion(suggest_path)

        st.image(suggest_path, caption="📱 Uploaded UI Screenshot", use_column_width=True)
        st.markdown("### ✨ Suggestions to Improve UI")
        st.code(suggestions)
