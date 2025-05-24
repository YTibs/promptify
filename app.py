import gradio as gr
import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch

# Load summarization pipeline once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_with_bart(captions):
    # Join captions into a paragraph
    text = " ".join(captions)

    # Hugging Face models have token limits; we truncate to ~1024 chars
    text = text[:1024]

    summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

# Load BLIP model once when the app starts
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def process_video(video_path, num_frames):
    os.makedirs("frames", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // (num_frames + 1), 1)

    extracted_frames = []
    captions = []

    for i in range(1, num_frames + 1):
        frame_number = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if not success:
            continue

        frame_path = f"frames/frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)
        extracted_frames.append(frame_path)

        # Generate caption for each frame
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)

    cap.release()

    # Combine captions into a paragraph
    summary = summarize_with_bart(captions)

    return extracted_frames[0], summary

demo = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload a short video"),
        gr.Slider(2, 10, value=4, step=1, label="Number of frames to extract")
    ],
    outputs=[
        gr.Image(type="filepath", label="Extracted Frame"),
        gr.Textbox(label="AI Description of the Task")
    ],
    title="Promptify",
    description="Upload a short video of a task. We'll extract a frame and describe what you're doing."
)

demo.launch()