import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torchvision

model = torchvision.models.efficientnet_b0()

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, out_features=3, bias=True)
)

model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(img):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()
    img = auto_transforms(image).unsqueeze(0)

    with torch.inference_mode():
        output = model(img)
        prediction = torch.softmax(output,dim=1).argmax(dim=1).item()
    return prediction

st.set_page_config(
    page_title="Tomato Leaf Health Checker",
    page_icon="🍅",
    layout="centered"
)

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:bold;
color:#2E7D32;
text-align:center;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
}

.result{
font-size:28px;
font-weight:bold;
text-align:center;
}

.tipbox{
background-color:#0047AB;
padding:20px;
border-radius:12px;
margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🍅 Tomato Leaf Health Detector</p>', unsafe_allow_html=True)

st.markdown("""
<p class="subtitle">
This AI application can analyze a tomato leaf image and predict the health condition of the plant.
Upload a tomato leaf photo and the system will determine whether the leaf is healthy
or affected by a disease.
</p>
""", unsafe_allow_html=True)


st.write("### 📷 Upload Tomato Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

st.markdown('</div>', unsafe_allow_html=True)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf Image", use_container_width=True,width=200)

    if st.button("🔍 Analyze Leaf"):

        pred = predict(image)

        label_map = {
            0: "Early Blight",
            1: "Late Blight",
            2: "Healthy"
        }

        result = label_map[pred]

        st.markdown(
            f'<p class="result">Prediction: {result}</p>',
            unsafe_allow_html=True
        )

        if pred == 0:

            st.markdown("""
<div class="tipbox">

### 🌿 Early Blight Tips

- Remove infected leaves immediately  
- Use fungicide spray  
- Avoid watering the leaves  
- Maintain proper spacing between plants  

</div>
""", unsafe_allow_html=True)


        elif pred == 1:

            st.markdown("""
<div class="tipbox">

### 🌿 Late Blight Tips

- Remove infected plants quickly  
- Use copper-based fungicide  
- Improve air circulation around plants  
- Reduce excess moisture in soil  

</div>
""", unsafe_allow_html=True)


        else:

            st.markdown("""
<div class="tipbox">

### ✅ Healthy Plant

Your tomato plant looks healthy.

Tips to keep it healthy:

- Maintain regular watering  
- Provide enough sunlight  
- Use balanced fertilizer  
- Check plants regularly for pests  

</div>
""", unsafe_allow_html=True)

st.write("---")
st.caption("AI Powered Tomato Plant Disease Detection 🍅")