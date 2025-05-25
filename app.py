import gradio as gr
import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
import csv
from datetime import datetime

# Log feedback to a CSV file
def log_feedback(summary, rating):
    with open("feedback_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), summary, rating])

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

    return extracted_frames, summary

# Handle feedback from the user
def handle_feedback(summary, feedback):
    log_feedback(summary, feedback)
    return f"Thanks for your feedback: {feedback}"

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Promptify\nTurn short videos into smart summaries using AI. Upload your video and we'll do the rest.")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload a short video", height=240)

    frame_slider = gr.Slider(2, 10, value=4, step=1, label="Number of frames to extract")
    submit_btn = gr.Button("Submit")

    extracted_frame = gr.Gallery(label="Extracted Frames")
    summary_output = gr.Textbox(label="AI Description of the Task")

    with gr.Row():
        good_btn = gr.Button("👍 Good")
        bad_btn = gr.Button("👎 Bad")
    feedback_msg = gr.Textbox(label="Feedback Result", interactive=False)

    submit_btn.click(
        fn=process_video,
        inputs=[video_input, frame_slider],
        outputs=[extracted_frame, summary_output]
    )

    good_btn.click(fn=handle_feedback, inputs=[summary_output, gr.Textbox(value="Good", visible=False)], outputs=feedback_msg)
    bad_btn.click(fn=handle_feedback, inputs=[summary_output, gr.Textbox(value="Bad", visible=False)], outputs=feedback_msg)

demo.launch()