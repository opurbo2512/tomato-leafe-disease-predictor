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


st.header("🍅 Tomato Leaf Health Detector")

with st.expander("About this app"):
    st.info("""
    This AI application can analyze a tomato leaf image and predict the health condition of the plant.
    Upload a tomato leaf photo and the system will determine whether the leaf is healthy
    or affected by a disease.
    """)


st.subheader("📷 Upload Tomato Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf Image", width=200)

    if st.button("🔍 Analyze Leaf"):

        pred = predict(image)

        label_map = {
            0: "Early Blight",
            1: "Late Blight",
            2: "Healthy"
        }

        result = label_map[pred]

        st.write(f"Result is : {label_map[pred]}")

        if pred == 0:

            st.error("""

### 🌿 Early Blight Tips

* Remove infected leaves immediately  
* Use fungicide spray  
* Avoid watering the leaves  
* Maintain proper spacing between plants
                          
""")


        elif pred == 1:

            st.error("""

### 🌿 Late Blight Tips

* Remove infected plants quickly  
* Use copper-based fungicide  
* Improve air circulation around plants  
* Reduce excess moisture in soil  
""")


        else:

            st.success("""

### ✅ Healthy Plant

Your tomato plant looks healthy.

Tips to keep it healthy:

* Maintain regular watering  
* Provide enough sunlight  
* Use balanced fertilizer  
* Check plants regularly for pests  

""")

st.write("---")
st.caption("AI Powered Tomato Plant Disease Detection 🍅")
