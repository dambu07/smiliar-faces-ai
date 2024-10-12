import numpy as np
import streamlit as st
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
from torchvision import transforms
import cv2

st.title("👨🏽‍💻 AI-Powered Face similarity search")
# st.subheader("Developed by Raavi with ❤️")
# st.caption("Please note: This app is only an estimation and is not a replacement in any way to human work.")
st.subheader("Upload 2 facial images and check if it is similar or not")
uploaded_file1 = st.file_uploader("Choose an image...", type=["jpg", "png", "gif"])
uploaded_file2 = st.file_uploader("Choose another image...", type=["jpg", "png", "gif"])

if uploaded_file1 is not None and uploaded_file2 is not None:
    img1 = Image.open(uploaded_file1).convert('RGB')
    img2 = Image.open(uploaded_file2).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption='Uploaded Image 1.', use_column_width=True)
    with col2:
        st.image(img2, caption='Uploaded Image 2.', use_column_width=True)

    st.write("Classifying...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def get_embedding(img):
        img = img.convert('RGB')  # Convert image to RGB
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        if boxes is None:
            st.error("No faces found in the image.")
            return None
        box = boxes[0]
        face = img.crop((box[0], box[1], box[2], box[3]))
        landmarks = landmarks[0]  # assuming only one face is detected

        # Get the eye coordinates
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Calculate the angle of rotation
        eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        # Get the bounding box size
        w, h = face.size

        # Get the center coordinates of the face
        center = (w // 2, h // 2)

        # Create the rotation matrix
        M = cv2.getRotationMatrix2D(center, eye_angle, 1)

        # Perform the rotation
        aligned_face = cv2.warpAffine(np.array(face), M, (w, h))

        # Preprocess the aligned face
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        face_tensor = preprocess(Image.fromarray(aligned_face)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face_tensor)
        return embedding

    embedding1 = get_embedding(img1)
    embedding2 = get_embedding(img2)

    if embedding1 is not None and embedding2 is not None:
        cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        st.write(f'Confidence: {cosine_similarity.item():.2f}')

        # You may want to adjust the threshold based on your requirements
        if cosine_similarity.item() > 0.55:
            st.success("The faces are similar.")
        else:
            st.error("The faces are not similar.")

footer = """
<style>
a:link, a:visited {
    color: #FFFFFF;
    background-color: transparent;
    text-decoration: underline;
}
a:hover, a:active {
    color: #7AE7C7;
    background-color: transparent;
    text-decoration: underline;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #F5BF03;
    text-align: center;
}
</style>
<div class="footer">
    <h6 style='text-align: center;'>
        <span style='color: #F81F6F;'>Developed with </span> 
        <span style='color: #f5f8fc;'> ❤ by Raavi </span>
    </h6>
</div>
"""
# Now, you can use this string in Streamlit's markdown
st.markdown(footer, unsafe_allow_html=True)
