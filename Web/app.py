import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Apply basic Streamlit configurations
st.set_page_config(page_title="Deep Learning Models", layout="wide")
st.title("Deep Learning Model Inference")
st.markdown("Upload data and run inference on models trained from Folders 4 to 8.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Model Definitions
# ==========================================

# Folder 4: ANN_4
class ANN_4(nn.Module):
    def __init__(self):
        super(ANN_4, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Folder 5: ANN (MNIST)
class ANN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Folder 6: CNN (MNIST)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.fc1(x)
        return x

# Folder 7: CNN (CatDog)
class CatDog_CNN(nn.Module):
    def __init__(self):
        super(CatDog_CNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.block5 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# Folder 7: CNN (CIFAR-10)
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256 * 2 * 2, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 10))
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

# Folder 7: CNN (PlantVillage)
class PlantVillage_CNN(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantVillage_CNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2))
        self.block5 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# Folder 8: RNNModel (Time Series)
class RNNModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1, num_layers=1, dropout=0.0):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ==========================================
# 2. Loading Models
# ==========================================

@st.cache_resource
def load_models():
    models = {}
    
    try:
        m4 = ANN_4().to(device)
        m4.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'ann_homework_model.pth'), map_location=device, weights_only=True))
        m4.eval()
        models['ann_4'] = m4
    except Exception as e:
        models['ann_4'] = f"Error loading ANN_4: {e}"

    try:
        m5 = ANN().to(device)
        m5.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'btvn_02_model.pth'), map_location=device, weights_only=True))
        m5.eval()
        models['ann_mnist'] = m5
    except Exception as e:
        models['ann_mnist'] = f"Error loading ANN MNIST: {e}"

    try:
        m6 = MNIST_CNN().to(device)
        m6.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'cnn_bt_model.pth'), map_location=device, weights_only=True))
        m6.eval()
        models['cnn_mnist'] = m6
    except Exception as e:
        models['cnn_mnist'] = f"Error loading CNN MNIST: {e}"

    try:
        m7_catdog = CatDog_CNN().to(device)
        m7_catdog.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'best_catdog_cnn.pth'), map_location=device, weights_only=True))
        m7_catdog.eval()
        models['cnn_catdog'] = m7_catdog
    except Exception as e:
        models['cnn_catdog'] = f"Error loading Cat/Dog CNN: {e}"

    try:
        m7_cifar = CIFAR10_CNN().to(device)
        m7_cifar.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'best_cifar10_cnn.pth'), map_location=device, weights_only=True))
        m7_cifar.eval()
        models['cnn_cifar'] = m7_cifar
    except Exception as e:
        models['cnn_cifar'] = f"Error loading CIFAR-10 CNN: {e}"

    try:
        m7_plant = PlantVillage_CNN(num_classes=38).to(device)
        m7_plant.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'best_plantvillage_cnn.pth'), map_location=device, weights_only=True))
        m7_plant.eval()
        models['cnn_plant'] = m7_plant
    except Exception as e:
        models['cnn_plant'] = f"Error loading PlantVillage CNN: {e}"

    try:
        m8 = RNNModel(input_size=3, hidden_size=32, output_size=1).to(device)
        m8.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'rnn_bt_model.pth'), map_location=device, weights_only=True))
        m8.eval()
        models['rnn'] = m8
    except Exception as e:
        models['rnn'] = f"Error loading RNN Model: {e}"

    return models

models = load_models()


# ==========================================
# 3. Streamlit UI & Inference
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Folder 4: ANN", 
    "Folder 5: MNIST ANN", 
    "Folder 6: MNIST CNN", 
    "Folder 7: CatDog", 
    "Folder 7: CIFAR-10",
    "Folder 7: PlantVillage",
    "Folder 8: RNN"
])

with tab1:
    st.header("Folder 4: Simple ANN (Tabular Data)")
    if isinstance(models['ann_4'], str):
        st.error(models['ann_4'])
    else:
        st.write("Enter 2 numeric features for binary classification:")
        f1 = st.number_input("Feature 1", value=0.0)
        f2 = st.number_input("Feature 2", value=0.0)
        if st.button("Predict Tabular", key='b1'):
            input_tensor = torch.tensor([[f1, f2]], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = models['ann_4'](input_tensor)
                prob = output.item()
            st.success(f"Predicted Probability: {prob:.4f}")
            st.info(f"Class: {1 if prob >= 0.5 else 0}")


def process_mnist(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0).to(device)


with tab2:
    st.header("Folder 5: ANN (MNIST)")
    if isinstance(models['ann_mnist'], str):
        st.error(models['ann_mnist'])
    else:
        img_file = st.file_uploader("Upload Digit Image", type=['png', 'jpg', 'jpeg'], key='u2')
        if img_file is not None:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", width=150)
            if st.button("Predict Digit (ANN)", key='b2'):
                input_tensor = process_mnist(image)
                with torch.no_grad():
                    output = models['ann_mnist'](input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                st.success(f"Predicted Digit: **{pred}**")


with tab3:
    st.header("Folder 6: CNN (MNIST)")
    if isinstance(models['cnn_mnist'], str):
        st.error(models['cnn_mnist'])
    else:
        img_file2 = st.file_uploader("Upload Digit Image", type=['png', 'jpg', 'jpeg'], key='u3')
        if img_file2 is not None:
            image2 = Image.open(img_file2)
            st.image(image2, caption="Uploaded Image", width=150)
            if st.button("Predict Digit (CNN)", key='b3'):
                input_tensor = process_mnist(image2)
                with torch.no_grad():
                    output = models['cnn_mnist'](input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                st.success(f"Predicted Digit: **{pred}**")


with tab4:
    st.header("Folder 7: CNN (Cat vs Dog)")
    if isinstance(models['cnn_catdog'], str):
        st.error(models['cnn_catdog'])
    else:
        img_file3 = st.file_uploader("Upload Cat or Dog Image", type=['png', 'jpg', 'jpeg'], key='u4')
        if img_file3 is not None:
            image3 = Image.open(img_file3)
            st.image(image3, caption="Uploaded Image", width=300)
            if st.button("Predict Pet", key='b4'):
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                if image3.mode != 'RGB':
                    image3 = image3.convert('RGB')
                input_tensor = transform(image3).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = models['cnn_catdog'](input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                classes = ['Cat', 'Dog']
                st.success(f"Prediction: **{classes[pred]}**")

with tab5:
    st.header("Folder 7: CNN (CIFAR-10)")
    if isinstance(models['cnn_cifar'], str):
        st.error(models['cnn_cifar'])
    else:
        img_file_cifar = st.file_uploader("Upload CIFAR-10 Object Image", type=['png', 'jpg', 'jpeg'], key='u5')
        if img_file_cifar is not None:
            image_cifar = Image.open(img_file_cifar)
            st.image(image_cifar, caption="Uploaded Image", width=300)
            if st.button("Predict CIFAR-10 Object", key='b5_cifar'):
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
                ])
                if image_cifar.mode != 'RGB':
                    image_cifar = image_cifar.convert('RGB')
                input_tensor = transform(image_cifar).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = models['cnn_cifar'](input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                
                cifar_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
                st.success(f"Prediction: **{cifar_classes[pred]}**")

with tab6:
    st.header("Folder 7: CNN (PlantVillage)")
    if isinstance(models['cnn_plant'], str):
        st.error(models['cnn_plant'])
    else:
        img_file_plant = st.file_uploader("Upload Plant Leaf Image", type=['png', 'jpg', 'jpeg'], key='u6')
        if img_file_plant is not None:
            image_plant = Image.open(img_file_plant)
            st.image(image_plant, caption="Uploaded Leaf Image", width=300)
            if st.button("Predict Disease", key='b6_plant'):
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                if image_plant.mode != 'RGB':
                    image_plant = image_plant.convert('RGB')
                input_tensor = transform(image_plant).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = models['cnn_plant'](input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                st.success(f"Prediction: **Disease Class #{pred} (out of 38)**")

with tab7:
    st.header("Folder 8: RNN (Time Series)")
    if isinstance(models['rnn'], str):
        st.error(models['rnn'])
    else:
        st.write("Enter an sequence of 3 features (Input Size = 3) for inference:")
        st.text("Example: Comma-separated feature values for the last time step.")
        seq_input = st.text_input("Enter 3 numbers, comma separated (e.g., 0.5, 0.1, 0.9):", key='i7')
        if st.button("Predict Next Step", key='b7'):
            try:
                features = [float(x.strip()) for x in seq_input.split(',')]
                if len(features) != 3:
                     st.error("Please enter exactly 3 numbers.")
                else:
                    input_tensor = torch.tensor([[features]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        output = models['rnn'](input_tensor)
                        pred_val = output.item()
                    st.success(f"Predicted Output (Scaled): {pred_val:.4f}")
            except Exception as e:
                st.error(f"Invalid input. Exception: {e}")
