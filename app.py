import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import requests
from io import BytesIO
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from openai import OpenAI

import plotly.graph_objects as go

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

import tempfile

# -----------------------------
# CUSTOM U-NET TREE MODEL
# -----------------------------
from predict import predict_tree_cover
# -----------------------------
# CONFIG
# -----------------------------

GROQ_API_KEY = "ENTER YOUR API KEY"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(
    page_title="EcoTwin AI",
    page_icon="🌍",
    layout="wide"
)
# -----------------------------
# SPLASH SCREEN
# -----------------------------

splash = st.empty()

splash.markdown("""
<div style="
height:90vh;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
background:#000000;
">

<h3 style="
color:white;
font-size:22px;
animation: blink 1s infinite;
">
Loading...
</h3>

<h1 style="
font-size:72px;
color:#00ff88;
text-shadow:
0 0 10px #00ff88,
0 0 20px #00ff88,
0 0 40px #00ff88;
">
🌍 EcoTwin AI
</h1>

<h3 style="
color:#ffffff;
letter-spacing:4px;
">
Powered by Neural Ravens
</h3>

<p style="
color:#cccccc;
font-size:18px;
margin-top:20px;
">
Analyzing Earth • Predicting Sustainability • Building a Greener Future
</p>

</div>
""", unsafe_allow_html=True)

time.sleep(3)

splash.empty()
# -----------------------------
# LOAD DOCUMENTS
# -----------------------------

@st.cache_resource
def load_rag():

    files = [
        "docs/trees.txt",
        "docs/solar.txt",
        "docs/waste.txt",
        "docs/green_cover.txt"
    ]

    documents = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            documents.append(f.read())

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embed_model.encode(documents)

    return documents, embed_model, embeddings

documents, embed_model, embeddings = load_rag()

# -----------------------------
# RETRIEVER
# -----------------------------

def retrieve_context(query):

    query_embedding = embed_model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )

    idx = np.argmax(scores)

    return documents[idx]

# -----------------------------
# LLM
# -----------------------------

def ask_ecotwin(question):

    context = retrieve_context(question)

    prompt = f"""
You are EcoTwin AI.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
    
def create_pdf(report_text):

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    doc = SimpleDocTemplate(
        temp_file.name
    )

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "EcoTwin AI Sustainability Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1,12)
    )

    content.append(
        Paragraph(
            report_text.replace("\n","<br/>"),
            styles["BodyText"]
        )
    )

    doc.build(content)

    return temp_file.name
# -----------------------------
# AI TREE CANOPY ANALYSIS
# USING CUSTOM U-NET MODEL
# -----------------------------

def analyze_green_cover(image):

    result, green_cover, pred_mask = predict_tree_cover(image)

    eco_score = min(
        100,
        int(green_cover * 2)
    )

    if green_cover < 10:
        green_index = "Very Low"
    elif green_cover < 20:
        green_index = "Low"
    elif green_cover < 35:
        green_index = "Moderate"
    elif green_cover < 50:
        green_index = "Good"
    else:
        green_index = "Excellent"

    return (
        green_cover,
        eco_score,
        green_index,
        result,
        pred_mask
    )
# -----------------------------
# PLANTATION ZONE DETECTION
# -----------------------------

def detect_plantation_zones(pred_mask):

    empty_area = (pred_mask == 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        empty_area,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    zones = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        x, y, w, h = cv2.boundingRect(cnt)

        ratio = w / h

        if (
            area > 5000
            and area < 25000
            and 0.5 < ratio < 2.5
        ):
            zones.append((x, y, w, h))

    return zones


# -----------------------------
# SATELLITE IMAGE FUNCTIONS
# -----------------------------

def deg2num(lat_deg, lon_deg, zoom):

    lat_rad = math.radians(lat_deg)

    n = 2.0 ** zoom

    xtile = int(
        (lon_deg + 180.0) / 360.0 * n
    )

    ytile = int(
        (
            1.0
            - math.asinh(
                math.tan(lat_rad)
            ) / math.pi
        )
        / 2.0 * n
    )

    return xtile, ytile


def get_satellite_image(lat, lon):

    zoom = 18

    x, y = deg2num(
        lat,
        lon,
        zoom
    )

    url = (
        f"https://server.arcgisonline.com/"
        f"ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/"
        f"{zoom}/{y}/{x}"
    )

    response = requests.get(url)

    image = Image.open(
        BytesIO(response.content)
    )

    image = image.convert("RGB")

    image = image.resize((640, 640))

    return image
# -----------------------------
# SIDEBAR
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Green Cover Analysis",
        "AI Assistant",
        "Report Generator"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "### 🧠 Powered by Neural Ravens"
)
# -----------------------------
# DASHBOARD
# -----------------------------
if menu == "Dashboard":

    # HERO SECTION
    st.markdown("""
    <div style='text-align:center;padding:20px;'>

    <h1 style='
    color:#00ff88;
    font-size:70px;
    font-weight:bold;
    text-shadow:
    0 0 10px #00ff88,
    0 0 20px #00ff88,
    0 0 40px #00ff88;
    '>
    🌍 EcoTwin AI
    </h1>

    <h3 style='color:#bdbdbd;'>
    AI Powered Sustainability Assessment Platform
    </h3>

    <p style='color:#888888;font-size:18px;'>
    Analyze • Predict • Sustain
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # METRICS
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🌳 Green Cover",
            "-- %",
            "Live Analysis"
        )

    with c2:
        st.metric(
            "🌍 Eco Score",
            "--",
            "AI Generated"
        )

    with c3:
        st.metric(
            "♻ Sustainability",
            "Ready",
            "System Online"
        )

    with c4:
        st.metric(
            "🤖 AI Assistant",
            "Active",
            "Groq Connected"
        )

    st.markdown("---")

    # INFO BANNER
    st.info(
        "🚀 Upload campus or satellite images to analyze green cover, eco score and plantation opportunities."
    )

    # FEATURE CARDS
    st.markdown("## 🌎 Core Features")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.success("""
### 🌳 Green Cover Analysis

AI-powered U-Net segmentation

✔ Tree Cover Detection

✔ Vegetation Mapping

✔ Green Area Assessment
""")

    with f2:
        st.success("""
### 🛰 Satellite Intelligence

Analyze locations using coordinates

✔ Satellite Imagery

✔ Lat/Lon Analysis

✔ Plantation Planning
""")

    with f3:
        st.success("""
### 🤖 Sustainability AI

Environmental insights & reports

✔ AI Assistant

✔ PDF Reports

✔ Eco Recommendations
""")

    st.markdown("---")

    # BOTTOM SECTION
    left, right = st.columns([2,1])

    with left:

        st.subheader("📊 Eco Health Index")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=78,
            title={'text':"Campus Sustainability"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range':[0,40],'color':'darkred'},
                    {'range':[40,70],'color':'orange'},
                    {'range':[70,100],'color':'darkgreen'}
                ]
            }
        ))

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color':"white"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("""
## 📌 Quick Stats

🌳 Tree Detection

🛰 Satellite Analysis

📄 PDF Reports

🤖 AI Assistant

🌍 Eco Score Prediction

🌱 Plantation Zones

♻ Sustainability Index
""")

        st.success(
            "System Status: Online"
        )

# may absorb approximately {carbon_absorption:,} kg of CO₂ per year."
# -----------------------------
# GREEN COVER ANALYSIS
# -----------------------------

elif menu == "Green Cover Analysis":

    st.header("🌳 Green Cover Analysis")

    method = st.radio(
        "Choose Input",
        [
            "Upload Image",
            "Latitude & Longitude"
        ]
    )

    image = None

    # Upload Image
    if method == "Upload Image":

        uploaded = st.file_uploader(
            "Upload Campus Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded)

    # Lat Lon Input
    else:

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude",
                value=22.5770,
                format="%.6f"
            )

        with col2:
            longitude = st.number_input(
                "Longitude",
                value=88.4795,
                format="%.6f"
            )

        if st.button("🌍 Analyze Location"):

            image = get_satellite_image(
                latitude,
                longitude
            )

    # Analysis
    if image is not None:

        (
            green_cover,
            eco_score,
            green_index,
            overlay,
            pred_mask
        ) = analyze_green_cover(image)

        zones = detect_plantation_zones(
            pred_mask
        )

        zone_img = overlay.copy()

        for i, (x, y, w, h) in enumerate(zones):

            cv2.rectangle(
                zone_img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3
            )

            cv2.putText(
                zone_img,
                f"Zone {i+1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        st.subheader("🌿 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Input Image",
                use_container_width=True
            )

        with col2:
            st.image(
                zone_img,
                caption="Tree Detection + Plantation Zones",
                use_container_width=True
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Tree Cover %",
            f"{green_cover:.2f}%"
        )

        c2.metric(
            "Eco Score",
            eco_score
        )

        c3.metric(
            "Green Index",
            green_index
        )

        st.subheader(
            "🌱 Recommended Plantation Zones"
        )

        if len(zones) == 0:

            st.warning(
                "No plantation zones detected."
            )

        else:

            for i in range(len(zones)):

                st.success(
                    f"Zone {i+1}: Suitable for tree plantation."
                )
    
       
        # -----------------------------
        # PLANTATION ZONE DETECTION
        # -----------------------------

        def detect_plantation_zones(pred_mask):
        
            empty_area = (pred_mask == 0).astype(np.uint8) * 255
        
            contours, _ = cv2.findContours(
                empty_area,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        
            zones = []
        
            for cnt in contours:
        
                area = cv2.contourArea(cnt)
        
                x, y, w, h = cv2.boundingRect(cnt)
        
                ratio = w / h
        
                if (
                    area > 5000
                    and area < 25000
                    and 0.5 < ratio < 2.5
                ):
                    zones.append((x, y, w, h))
        
            return zones
    
    
        def deg2num(lat_deg, lon_deg, zoom):

            lat_rad = math.radians(lat_deg)
        
            n = 2.0 ** zoom
            
            xtile = int(
                (lon_deg + 180.0) / 360.0 * n
            )
            
            ytile = int(
                (
                    1.0
                    - math.asinh(
                        math.tan(lat_rad)
                    ) / math.pi
                )
                / 2.0 * n
            )
            
            return xtile, ytile

        # -----------------------------
        # FREE ESRI SATELLITE IMAGE
        # -----------------------------
        
        def deg2num(lat_deg, lon_deg, zoom):
        
            lat_rad = math.radians(lat_deg)
        
            n = 2.0 ** zoom
        
            xtile = int(
                (lon_deg + 180.0) / 360.0 * n
            )
        
            ytile = int(
                (
                    1.0
                    - math.asinh(
                        math.tan(lat_rad)
                    ) / math.pi
                )
                / 2.0 * n
            )
        
            return xtile, ytile
        
        


        # -----------------------------
        # METRICS
        # -----------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Tree Cover %",
            f"{green_cover:.2f}"
        )

        c2.metric(
            "Eco Score",
            eco_score
        )

        c3.metric(
            "Index",
            green_index
        )
        # -----------------
        # U-NET TREE OVERLAY
        # -----------------

        zone_img = overlay.copy()

        for i, (x, y, w, h) in enumerate(zones):
        
            cv2.rectangle(
                zone_img,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                3
            )
        
            cv2.putText(
                zone_img,
                f"Zone {i+1}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
        
        blended = zone_img

        # -----------------
        # DRAW PLANTATION ZONES
        # -----------------
        
        zone_img = blended.copy()
        
        for i, (x, y, w, h) in enumerate(zones):
        
            cv2.rectangle(
                zone_img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3
            )
        
            cv2.putText(
                zone_img,
                f"Zone {i+1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
        st.subheader("🌿 AI Green Cover Detection")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original Campus Image",
                use_container_width=True
            )

        with col2:

            st.image(
                zone_img,
                caption="🌱 Suggested Plantation Zones",
                use_container_width=True
            )
        # -----------------
        # PLANTATION RECOMMENDATIONS
        # -----------------
        
        st.subheader("🌱 Recommended Plantation Zones")
        
        if len(zones) == 0:
        
            st.warning(
                "No large plantation zones detected."
            )
        
        else:
        
            for i, zone in enumerate(zones):
        
                st.success(
                    f"Zone {i+1}: Suitable for additional tree plantation."
                )       


        # -----------------
        # METRICS
        # -----------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Green Cover %",
            f"{green_cover:.2f}"
        )

        c2.metric(
            "Eco Score",
            eco_score
        )

        c3.metric(
            "Index",
            green_index
        )

        # -----------------
        # GAUGE CHART
        # -----------------

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eco_score,
            title={'text': "Eco Score"},
            gauge={
                'axis': {
                    'range': [0, 100]
                }
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------
        # TREE SIMULATOR
        # -----------------

        st.subheader(
            "🌱 Tree Plantation Simulator"
        )

        trees = st.slider(
            "Additional Trees",
            0,
            500,
            50
        )

        predicted_cover = min(
            green_cover + trees * 0.05,
            100
        )

        predicted_score = min(
            int(predicted_cover * 2),
            100
        )

        c4, c5 = st.columns(2)

        c4.metric(
            "Predicted Green Cover",
            f"{predicted_cover:.2f}%"
        )

        c5.metric(
            "Predicted Eco Score",
            predicted_score
        )
        # -----------------
        # SUSTAINABILITY
        # PREDICTION GRAPH
        # -----------------
        
        import pandas as pd
        
        graph_data = pd.DataFrame({
        
            "Trees": [
                0,
                50,
                100,
                200,
                300,
                500
            ],
        
            "Eco Score": [
        
                eco_score,
        
                min(
                    100,
                    int((green_cover + 2.5) * 2)
                ),
        
                min(
                    100,
                    int((green_cover + 5) * 2)
                ),
        
                min(
                    100,
                    int((green_cover + 10) * 2)
                ),
        
                min(
                    100,
                    int((green_cover + 15) * 2)
                ),
        
                min(
                    100,
                    int((green_cover + 25) * 2)
                )
            ]
        })
        
        st.subheader(
            "📈 Sustainability Prediction"
        )
        
        st.caption(
            "Projected Eco Score after additional tree plantation."
        )
        
        st.line_chart(
            graph_data.set_index(
                "Trees"
            )
        )
        # -----------------
        # CARBON IMPACT
        # -----------------

        st.subheader(
            "🌎 Carbon Impact Estimation"
        )

        carbon_absorption = trees * 22

        st.metric(
            "Estimated Carbon Absorption (kg/year)",
            f"{carbon_absorption:,}"
        )

        st.info(
            f"Planting {trees} trees may absorb approximately {carbon_absorption:,} kg of CO₂ per year."
        )
# -----------------------------
# AI ASSISTANT
# -----------------------------

elif menu == "AI Assistant":

    import time

    st.header("🤖 EcoTwin Sustainability Assistant")

    st.caption(
        "Ask about sustainability, green cover, solar energy, waste management, and campus planning."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show old messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"],
            avatar="🌍" if message["role"] == "assistant" else "👨‍💻"
        ):
            st.markdown(message["content"])

    # User Input

    prompt = st.chat_input(
        "Ask EcoTwin AI..."
    )

    if prompt:

        # Show user message

        with st.chat_message(
            "user",
            avatar="👨‍💻"
        ):
            st.markdown(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # Generate Answer

        answer = ask_ecotwin(prompt)

        # Assistant Message

        with st.chat_message(
            "assistant",
            avatar="🌍"
        ):

            placeholder = st.empty()

            placeholder.markdown(
                "🧠 Analyzing sustainability data...\n\n"
            )

            full_response = ""

            for word in answer.split():

                full_response += word + " "

                placeholder.markdown(
                    full_response + "▌"
                )

                time.sleep(0.02)

            placeholder.markdown(
                full_response
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# -----------------------------
# REPORT GENERATOR
# -----------------------------

elif menu == "Report Generator":

    st.header(
        "📄 Sustainability Report Generator"
    )

    uploaded = st.file_uploader(
        "Upload Campus Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        (
            green_cover,
            eco_score,
            green_index,
            overlay,
            pred_mask
         ) = analyze_green_cover(image)

        with st.spinner(
            "Generating AI Recommendations..."
        ):

            recommendation = ask_ecotwin(
                "How can our campus improve sustainability?"
            )

        heat_risk = (
            "High"
            if green_cover < 15
            else "Moderate"
        )

        report = f"""
Campus Sustainability Report

--------------------------------

Green Cover:
{green_cover:.2f} %

Eco Score:
{eco_score}/100

Green Cover Index:
{green_index}

Heat Risk:
{heat_risk}

--------------------------------

AI Recommendations:

{recommendation}

--------------------------------

Generated by EcoTwin AI
"""

        st.text_area(
            "Generated Report",
            report,
            height=350
        )
        pdf_path = create_pdf(report)

        with open(
            pdf_path,
            "rb"
        ) as pdf_file:
        
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_file,
                file_name="EcoTwin_Report.pdf",
                mime="application/pdf"
            )