import gradio as gr
import cv2
import os

def extract_frame(video_path):
    # Create a directory to store frame(s)
    os.makedirs("frames", exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    # Go to 1 second into video (adjust if needed)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    success, frame = cap.read()

    if success:
        frame_path = "frames/frame.jpg"
        cv2.imwrite(frame_path, frame)
        return frame_path
    else:
        return "Failed to extract frame."

demo = gr.Interface(
    fn=extract_frame,
    inputs=gr.Video(label="Upload a short video"),
    outputs=gr.Image(type="filepath", label="Extracted Frame"),
    title="Promptify My Workflow",
    description="Upload a short video of a task. We'll extract a frame to analyze what you did."
)

demo.launch()