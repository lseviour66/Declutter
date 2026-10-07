import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import random
import os
os.environ["YOLO_NO_CV2"] = "1"


# -------------------------------------------------
# GLOBAL MODELS ONLY
# -------------------------------------------------

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_yolo():
    return YOLO("yolov8m.pt")


# -------------------------------------------------
# CREATIVE POOLS (cleaned + expanded)
# -------------------------------------------------

CATEGORY_META = {
    "blankets": [
        ("Blankets / Throws", "🛋️", "#FFE4C4"),
        ("Cozy Culprits", "🧸", "#FFF2D7"),
        ("Soft Layer Squad", "🌙", "#FDEBD0"),
        ("Warm Chaos Fabrics", "🧶", "#F7EEDB"),
    ],
    "clothes": [
        ("Clothes", "👕", "#CFFFE5"),
        ("Wardrobe Escapees", "🧦", "#D9FFF1"),
        ("Fabric Rebels", "👚", "#E0FFF7"),
        ("Laundry Renegades", "🩳", "#C8F7E4"),
    ],
    "pillows": [
        ("Pillows", "🛏️", "#E6D6FF"),
        ("Soft Head Supporters", "🛌", "#F0E6FF"),
        ("Cushion Collective", "🧘", "#E9DFFF"),
        ("Comfort Overachievers", "🌤️", "#F5EFFF"),
    ],
    "cat": [
        ("Cat", "🐈", "#FFF7C2"),
        ("Furry Supervisor", "😼", "#FFF4D0"),
        ("Tiny Judgment Machine", "🐾", "#FFF8DD"),
        ("Domestic Overlord", "👑", "#FFF3C9"),
    ],
    "sofa": [
        ("Sofa", "🛋️", "#CFE9FF"),
        ("Comfy Command Center", "🪑", "#D9EEFF"),
        ("Seat of Power", "🛋️", "#CDE7FF"),
        ("Cushioned Throne", "🪑", "#D0F0FF"),
    ],
    "furniture": [
        ("Furniture", "🪑", "#D4FFF8"),
        ("Structural Support Squad", "📦", "#D9FFF4"),
        ("Home Infrastructure", "🛠️", "#CFFFEF"),
        ("Anti‑Chaos Framework", "🧱", "#D8FFF6"),
    ],
    "clutter": [
        ("Clutter", "🧹", "#FFD6E8"),
        ("Entropy Hotspot", "🔥", "#FFE1EE"),
        ("Chaos Cluster", "🎲", "#FFD9E2"),
        ("Messy Mischief", "🌀", "#FFE6F1"),
    ],
    "general": [
        ("General Room", "🏠", "#E8F0FF"),
        ("Everyday Space", "🧭", "#EEF4FF"),
        ("Life Zone", "🌿", "#E3EBFF"),
        ("Human Habitat", "🪴", "#EAF1FF"),
    ],
}

SNARK_POOL = {
    "blankets": [
        "These blankets appear to be forming a diplomatic summit.",
        "A cozy uprising is happening in blanket territory.",
        "Blankets lounging around like they pay rent.",
        "Your blankets are practicing interpretive dance.",
    ],
    "clothes": [
        "Your clothes are enjoying a brief sabbatical from the wardrobe.",
        "Fabric freedom fighters spotted.",
        "Laundry has declared independence.",
        "Your clothes are living their best unsupervised life.",
    ],
    "pillows": [
        "Your pillows are multiplying like soft rabbits.",
        "Pillows gathering for a secret council.",
        "These pillows are plotting a comfort coup.",
        "Pillows vibing with zero responsibilities.",
    ],
    "cat": [
        "Your cat is silently judging everything.",
        "A furry monarch surveys their kingdom.",
        "Your cat knows something you don’t.",
        "This feline is running the entire operation.",
    ],
    "sofa": [
        "Your sofa is carrying emotional and physical baggage.",
        "This sofa has seen things.",
        "The sofa is absorbing the chaos with grace.",
        "Your sofa is holding the fort like a champ.",
    ],
    "furniture": [
        "Your furniture is holding the line against total entropy.",
        "Furniture standing firm in the face of chaos.",
        "These furniture pieces are doing overtime.",
        "Home infrastructure working without complaint.",
    ],
    "clutter": [
        "Your clutter has achieved sentience.",
        "Chaos pockets detected.",
        "This clutter is auditioning for a reality show.",
        "Entropy is thriving here.",
    ],
    "general": [
        "This room has strong ‘life happens here’ energy.",
        "A space filled with stories and stuff.",
        "This room is vibing in its own unique way.",
        "A perfectly normal room doing its best.",
    ],
}

RECOMMENDATION_POOL = {
    "blankets": [
        "Fold or drape blankets to instantly calm the visual noise.",
        "Consider storing extra throws in a basket.",
        "One neatly placed blanket creates a cleaner aesthetic.",
    ],
    "clothes": [
        "Gather clothes into a hamper or designated spot.",
        "A quick fold-and-sort session will transform the space.",
        "Consider a small laundry station nearby.",
    ],
    "pillows": [
        "Stack or align pillows to restore order.",
        "Limit pillows to two or three for a cleaner look.",
        "Group pillows by color for a cohesive vibe.",
    ],
    "cat": [
        "Your cat approves of tidiness—clean around their favorite spots.",
        "Create a small cat zone to reduce spread.",
        "A tidy environment keeps your cat curious and happy.",
    ],
    "sofa": [
        "Clear items off the sofa to reclaim comfort space.",
        "A quick cushion realignment works wonders.",
        "Consider a throw to unify the sofa visually.",
    ],
    "furniture": [
        "Declutter surfaces to let the furniture shine.",
        "Group items logically on tables or shelves.",
        "A clean furniture layout improves flow.",
    ],
    "clutter": [
        "Start with one small area—momentum builds fast.",
        "Sort clutter into keep / toss / relocate piles.",
        "Use baskets or trays to contain loose items.",
    ],
    "general": [
        "A quick tidy session will brighten the whole room.",
        "Focus on one corner to create a sense of progress.",
        "Small changes can dramatically improve the vibe.",
    ],
}


# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------

def main():
    st.set_page_config(page_title="Declutter — Creative Edition", layout="wide")
    st.title("🧹 Declutter My Life Dashboard")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:

        img = Image.open(uploaded)
        img = ImageOps.exif_transpose(img).convert("RGB")

        processor, blip_model = load_blip()
        yolo_model = load_yolo()

        # -----------------------------
        # BLIP (dynamic prompts)
        # -----------------------------
        def blip_describe(img):
            question_pool = {
                "objects": [
                    "List the main objects you can identify.",
                    "What items are present in this scene?",
                    "Describe the key objects visible here.",
                ],
                "fabric": [
                    "List all fabric items separately.",
                    "Identify blankets, clothes, pillows, and other fabrics.",
                    "What fabric-based items do you see?",
                ],
                "animal": [
                    "Is there an animal in this image?",
                    "Identify any animals present.",
                    "Do you see a pet or creature here?",
                ],
                "furniture": [
                    "Describe the furniture in this room.",
                    "What furniture pieces are visible?",
                    "Identify chairs, tables, sofas, or other furniture.",
                ],
                "clutter": [
                    "Describe any clutter or mess.",
                    "Identify areas of disorder or scattered items.",
                    "What messy or chaotic elements do you see?",
                ],
            }

            answers = {}
            for key, qlist in question_pool.items():
                q = random.choice(qlist)
                inputs = processor(images=img, text=q, return_tensors="pt")
                with torch.no_grad():
                    out = blip_model.generate(**inputs, max_new_tokens=80)
                answers[key] = processor.decode(out[0], skip_special_tokens=True).lower()

            return answers

        blip = blip_describe(img)

        # -----------------------------
        # YOLO detection
        # -----------------------------
        def detect(img):
            results = yolo_model.predict(np.array(img), conf=0.10)
            dets = []
            for r in results:
                for box in r.boxes:
                    label = yolo_model.names[int(box.cls[0])].lower()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append({"label": label, "box": (x1, y1, x2, y2)})
            return dets

        detections = detect(img.copy())

        # -----------------------------
        # Category extraction
        # -----------------------------
        categories = set()

        fabric = blip["fabric"]
        if any(k in fabric for k in ["blanket", "throw"]):
            categories.add("blankets")
        if any(k in fabric for k in ["clothes", "shirt", "laundry"]):
            categories.add("clothes")
        if any(k in fabric for k in ["pillow", "cushion"]):
            categories.add("pillows")

        furn = blip["furniture"]
        if any(k in furn for k in ["sofa", "couch"]):
            categories.add("sofa")
        if any(k in furn for k in ["chair", "table", "shelf", "furniture"]):
            categories.add("furniture")

        cl = blip["clutter"]
        if any(k in cl for k in ["mess", "pile", "chaos", "objects", "stuff"]):
            categories.add("clutter")

        for det in detections:
            label = det["label"]
            if any(k in label for k in ["chair", "table", "shelf"]):
                categories.add("furniture")
            if any(k in label for k in ["bag", "backpack", "handbag", "suitcase", "cup", "bottle"]):
                categories.add("clutter")

        # -----------------------------
        # VALIDATION LAYER (Option C)
        # -----------------------------
        sofa_present = False

        if "sofa" in furn or "couch" in furn:
            sofa_present = True

        for det in detections:
            if det["label"] in ["sofa", "couch"]:
                sofa_present = True

        if not sofa_present and "sofa" in categories:
            categories.remove("sofa")

        if not categories:
            categories.add("general")

        # -----------------------------
        # Build tiles
        # -----------------------------
        tiles = []
        for cat in categories:
            label, emoji, color = random.choice(CATEGORY_META[cat])
            desc = random.choice(SNARK_POOL[cat])
            tiles.append((cat, label, emoji, color, desc))

        random.shuffle(tiles)

        # -----------------------------
        # Build recommendations
        # -----------------------------
        recommendations = []
        for cat in categories:
            rec = random.choice(RECOMMENDATION_POOL[cat])
            recommendations.append(f"- {rec}")

        # -----------------------------
        # YOLO overlay
        # -----------------------------
        def draw_boxes(img, dets):
            draw = ImageDraw.Draw(img)
            for det in dets:
                x1, y1, x2, y2 = det["box"]
                draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=3)
                draw.text((x1 + 3, y1 + 3), det["label"], fill="#FF0000")
            return img

        annotated = draw_boxes(img.copy(), detections)

        # -----------------------------
        # UI layout
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Step 1: Original Image")
            st.image(img, use_container_width=True)

        with col2:
            st.markdown("### Step 2: What the AI Sees")
            for _, label, emoji, color, desc in tiles:
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 14px;
                        padding: 14px;
                        margin-bottom: 14px;
                        background-color: {color};
                        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
                        text-align: center;
                    ">
                        <div style="font-size: 32px;">{emoji}</div>
                        <div style="font-weight: bold; font-size: 16px;">{label}</div>
                        <div style="font-size: 13px;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col3:
            st.markdown("### Step 3: Recommendations")
            st.write("\n".join(recommendations))

        st.markdown("---")
        st.markdown("### YOLO Detection Overlay")
        st.image(annotated, use_container_width=True)


if __name__ == "__main__":
    main()
