import gradio as gr

def handle_video(video):
    return "You uploaded a video! 🎥"

demo = gr.Interface(
    fn=handle_video,
    inputs=gr.Video(label="Upload your task video"),
    outputs="text",
    title="Promptify My Workflow",
    description="Upload a short video of a task, and we’ll turn it into a prompt."
)

demo.launch()