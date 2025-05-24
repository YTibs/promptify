import gradio as gr
import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model once when the app starts
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def process_video(video_path):
    os.makedirs("frames", exist_ok=True)

    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    success, frame = cap.read()
    
    if not success:
        return None, "Could not extract frame from video."

    frame_path = "frames/frame.jpg"
    cv2.imwrite(frame_path, frame)

    # Generate caption with BLIP
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return frame_path, caption

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a short video"),
    outputs=[
        gr.Image(type="filepath", label="Extracted Frame"),
        gr.Textbox(label="AI Description of the Task")
    ],
    title="Promptify",
    description="Upload a short video of a task. We'll extract a frame and describe what you're doing."
)

demo.launch()