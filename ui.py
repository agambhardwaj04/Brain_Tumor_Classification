import streamlit as st
import numpy as np
from PIL import Image
import streamlit as st
import joblib
from tensorflow.keras.models import load_model
model = load_model('brain_tumor_model.h5')
st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        padding: 6px;
        border-radius: 10px;
    }
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: white;
    }
    /* Sidebar title */
    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        color: #ff4b4b;
        margin-bottom: 10px;
    }
    /* Sidebar section headings */
    .sidebar-section {
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        color: #ffcc00;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .sidebar-logo {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .sidebar-logo img {
        border-radius: 50%;  /* Makes image circular */
        width: 80px;        /* Adjust size */
        height: 80px;
    }
    </style>
""", unsafe_allow_html=True)

# Set the app title
st.set_page_config(page_title="Brain Tumor Detection",page_icon="./Images/brain.png" ,layout="wide")

logo = Image.open("./Images/brain.png")  # Load the logo image
st.sidebar.image(
    logo,
    width=150,   # Adjust size (smaller width)
    caption="Brain Tumor Detection"
)

st.sidebar.markdown("---")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.title('🧠 Brain Tumor Detection App')
page = st.sidebar.selectbox("Go to", ["🌐 Home", "🔍 Detection", "📊 Dataset Analysis", "ℹ️ About", "📬 Contact"])
st.sidebar.markdown("\n")


st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-section">📊 Model Information</div>', unsafe_allow_html=True)
st.sidebar.info("Model: VGG19 (150x150x3 Input)\n,  Optimizer: Adam,\nLoss: Binary Crossentropy")
st.sidebar.markdown("---")

accuracy = 83.76  # Example accuracy
st.sidebar.progress(accuracy/100)
st.sidebar.write(f"Accuracy: {accuracy}%")

# CSS for styled image (for Home page)
st.markdown("""
    <style>
    .styled-image {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        width: 25%;
        height: 25%;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# Define each page
if page == "🌐 Home":
    st.title("Welcome to the Brain Tumor Detection App 🧠")
    st.header('Introduction')
    with st.expander('🧠 What is Brain Tumor ?'):
      st.write('A brain tumor is an abnormal growth of cells in the brain. It can be benign (non-cancerous) or malignant (cancerous). Brain tumors can cause various symptoms depending on their size and location, including headaches, seizures, and cognitive changes. Even benign tumors can be harmful because they may press against sensitive areas of the brain, affecting its normal function. Malignant tumors are more aggressive, growing quickly and often spreading into surrounding brain tissue. Since the brain controls all bodily activities, even a small tumor can have significant effects depending on where it is located. Brain tumors are diagnosed through imaging techniques such as MRI or CT scans and are often followed by a biopsy to confirm the type. Treatment typically involves a combination of surgery, radiation therapy, chemotherapy, or targeted drug therapy, depending on the tumor’s characteristics. Advances in technology and medical research have improved outcomes, especially when tumors are detected early.')
    with st.expander("🤒 Symptoms of Brain tumor"):
      st.write("""
      -> Persistent headaches, often worse in the morning or when lying down

      -> Nausea and vomiting, especially early in the day

      -> Seizures, even if there’s no prior history of epilepsy

      -> Loss of consciousness

      -> Vision problems – blurred or double vision, or loss of peripheral vision

      -> Hearing problems – ringing in the ears or partial hearing loss

      -> Difficulty with balance or coordination

      -> Muscle weakness or numbness, usually on one side of the body
      -> Seizures, even if there’s no prior history of epilepsy

      -> Vision problems – blurred or double vision, or loss of peripheral vision

      -> Hearing problems – ringing in the ears or partial hearing loss

      -> Difficulty with balance or coordination

      -> Muscle weakness or numbness, usually on one side of the body

      -> Speech difficulties – trouble speaking or understanding language

      -> Confusion, memory loss, or changes in personality or behavior""")

    with st.expander("🔬 What is Brain Tumor Detection ?"):
      st.write("""
        Brain tumor detection is a critical process in diagnosing and treating abnormal growths within the brain. It typically involves advanced imaging techniques such as MRI (Magnetic Resonance Imaging) or CT (Computed Tomography) scans, which provide detailed views of the brain’s structure. Accurate detection is essential for determining the presence, size, location, and type of tumor, all of which influence the treatment plan. In recent years, technology has greatly enhanced the detection process, with artificial intelligence and deep learning models increasingly being used to analyze medical images. These tools help radiologists identify tumors more quickly and accurately, even in early stages, reducing the chances of misdiagnosis. Early and precise detection not only improves treatment outcomes but also increases the chances of survival and recovery for patients. 
    """)

    with st.expander('🩻 How Deep Learning and AI is used in Brain Tumor Detection ?'):
      st.write("""Deep learning, a specialized area  of artificial intelligence (AI), plays a transformative role in brain tumor detection and medical imaging. Unlike traditional methods that rely heavily on manual feature extraction, deep learning uses neural networks—especially convolutional neural networks (CNNs)—to automatically learn and identify complex patterns within medical images such as MRI or CT scans. These models can be trained on thousands of annotated brain scans to distinguish between healthy tissue and tumors, or even to classify tumor types and grades.
      The integration of AI in healthcare offers several key advantages. Deep learning systems can analyze images with remarkable speed and consistency, significantly reducing the time required for diagnosis. They also help minimize human error by detecting subtle abnormalities that might be missed by the human eye. In brain tumor diagnosis, AI assists radiologists by highlighting potential problem areas, ranking tumor severity, and suggesting probable classifications. This augmented intelligence does not replace doctors but supports them in making more accurate, faster, and data-driven decisions.""")
    with st.expander('🌟 Benefits of Early Detection of Brain Tumors'):
       st.write('Early detection of brain tumors plays a crucial role in improving patient outcomes and survival rates. Identifying a tumor in its initial stages often allows for more effective and less invasive treatment options, reducing the risk of complications. It can help preserve vital brain functions such as memory, movement, and speech by preventing the tumor from growing or spreading. Early diagnosis also enables timely medical intervention, increases the chances of full recovery, and can significantly lower treatment costs. Overall, early detection empowers both patients and doctors to make informed decisions and plan a personalized care strategy')

    with st.expander('💻 How to use this Website ?'):
      st.write("""
      1. **Upload Image**: Click on the "Upload Image" button to select a brain MRI image from your device.
      2. **View Results**: After uploading, the app will process the image and display the results, indicating whether a tumor is detected and providing additional information.
      3. **Interpretation**: The results will tell you have a tumor present or not
      4. **Feedback**: You can provide feedback on the accuracy of the detection to help improve the model.
    """)
    with st.expander('💡 Example of the Image used for Brain Tumor Detection'):
           st.write("Below is an example of a brain MRI image used for tumor detection:")
           st.markdown("""
        <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-023-41576-6/MediaObjects/41598_2023_41576_Fig1_HTML.jpg" class="styled-image">
    """, unsafe_allow_html=True)
    with st.expander('⚠️ Disclaimer'):
       st.write('This application is intended for educational and research purposes only. It uses a machine learning model to assist in the detection of brain tumors from MRI images, but it is not a substitute for professional medical advice, diagnosis, or treatment. The results generated by this tool should not be used as the sole basis for making any healthcare decisions. Always consult a qualified healthcare provider or radiologist for an accurate diagnosis and appropriate medical guidance. The creators of this application are not responsible for any clinical decisions made based on its outputs. Use this tool responsibly and with the understanding that it is a supplementary resource in the field of medical imaging and diagnosis.')


elif page == "🔍 Detection":
    st.title("🧠 Brain Tumor Detection")
    st.subheader("📤 Upload an MRI scan below to detect brain tumors.")
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((150, 150))  # Match training input size
        img_array = np.array(image) / 255.0  # Normalize to 0-1
        img_array = np.expand_dims(img_array, axis=0)  # ✅ Shape: (1,150,150,3)

        st.write("Image successfully uploaded!")

        with st.spinner("🔄 Processing image... Please wait"):
            prediction = model.predict(img_array)[0][0]

        # Show prediction result
        if prediction > 0.5:
            st.error(f"🧠 Tumor Detected with confidence {prediction:.2f}")
        else:
            st.success(f"✅ No Tumor Detected with confidence {1-prediction:.2f}")

elif page == "📊 Dataset Analysis":
    st.title("📊 Dataset Analysis")
    st.write("""
        The dataset used for training this model consists of brain MRI images.
        It includes a variety of cases with and without tumors to ensure robust detection capabilities.
    """)
    st.write("The model has been trained to recognize patterns associated with brain tumors, improving its accuracy over time.")
    st.write("This dataset is taken from Kaggle and contains 4600 images of brain MRI scans, with 2087 images labeled as 'No Tumor' and 2513 images labeled as 'Tumor'.")
    st.write('Dataset Link: [Brain MRI Images for Brain Tumor Detection] https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset')
    st.image("./Images/dataset_number.png", caption="Example MRI Image from Dataset", use_container_width=True)
    st.image("./Images/data_analysis.png", caption="Example MRI Image from Dataset", use_container_width=True)
    st.image("./Images/ROC_Curve.png", caption="Example MRI Image from Dataset", use_container_width=True)


elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.write("""
        This application uses deep learning techniques to help detect brain tumors from MRI scans.
        It is designed to assist radiologists and improve diagnostic accuracy.
    """)

elif page == "📬 Contact":
    st.title("📬 Contact")
    st.write("*Please use this application responsibly and consult healthcare professionals for medical advice. This app is not a substitute for professional medical diagnosis or treatment, rather it is a tool to assist in the detection of brain tumors from MRI images.*")
    st.write("Thank you for using the Brain Tumor Detection App!")
    st.write("We hope you find it useful in your medical imaging and diagnostic tasks.")
    st.write("If you have any questions, suggestions, or feedback, feel free to reach out!")
    st.write("Contact us at: **bhardwajagam62@gmail.com**")

