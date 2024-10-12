import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import EmotionCNN  # Ensure your model class is in this file

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)
model.load_state_dict(torch.load('models/emotion_cnn.pth'))
model.eval()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Inference function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)  # Get the index of the max output
        predicted_label = emotion_labels[predicted_idx.item()]  # Get the label from the index
        return predicted_label  # Return the emotion label

# Create a Gradio interface for webcam input
iface = gr.Interface(fn=predict, 
                     inputs=gr.Image(type="pil", label="Take a Picture"), 
                     outputs=gr.Text(label="Predicted Emotion"),
                     live=True)

# Launch the interface
if __name__ == "__main__":
    iface.launch()