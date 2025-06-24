import os
import base64
import openai
import cv2
from skimage.metrics import structural_similarity as ssim



def compare_images(figma_path, app_path):
    img1 = cv2.imread(figma_path)
    img2 = cv2.imread(app_path)

    img1 = cv2.resize(img1, (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0])))
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    diff_path = os.path.join("backend/diffs", os.path.basename(app_path).replace(".png", "_diff.png"))
    cv2.imwrite(diff_path, img2)
    return diff_path, score

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_openai_cosmetic_diff(figma_path, app_path):
    figma_b64 = encode_image(figma_path)
    app_b64 = encode_image(app_path)

    response = openai.chat.completions.create(
    model="gpt-4-turbo",  # or "gpt-4o" for better image analysis
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You're an expert UI/UX visual tester. Compare the provided Figma UI design and the actual App screenshot.\n\n"
                        "Focus on identifying true **cosmetic bugs**, NOT those caused by differences in screen resolution or aspect ratio.\n\n"
                        "**Step-by-step Instructions:**\n"
                        "1. First, detect and list **missing UI components** (text, icons, buttons, labels, images, etc.) — this is the top priority.\n"
                        "2. Check for **spelling or text content errors**.\n"
                        "3. Highlight visual bugs that degrade the design consistency, such as:\n"
                        "- Color mismatches\n"
                        "- Font size or type inconsistencies\n"
                        "- Spacing or padding issues (but ignore if it's due to aspect ratio)\n"
                        "- Incorrect or disproportionate element sizes\n\n"
                        "⚠️ Ignore layout misalignments unless they significantly break design structure.\n"
                        "⚠️ Focus on **relative accuracy** over pixel-perfection.\n\n"
                        "Return the report in this format:\n"
                        "**Missing UI Elements:**\n"
                        "- ...\n\n"
                        "**Text Errors / Spelling Mistakes:**\n"
                        "- ...\n\n"
                        "**Other Cosmetic Differences:**\n"
                        "- ..."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{figma_b64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{app_b64}"}
                }
            ]
        }
    ],
    max_tokens=1000
)


    return response.choices[0].message.content

def ask_ui_suggestion(app_path):
    app_b64 = encode_image(app_path)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Suggest UI/UX improvements for this app screen."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{app_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=700
    )

    return response.choices[0].message.content
