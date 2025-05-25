import gradio as gr

def say_hi(name):
    return f"Hi {name}!"

def handle_feedback(summary, rating):
    return f"You marked this summary as: {rating}"

with gr.Blocks() as demo:
    name_input = gr.Textbox(label="Name")
    submit_btn = gr.Button("Say Hi")
    output = gr.Textbox(label="Greeting")

    with gr.Row():
        good_btn = gr.Button("👍 Good")
        bad_btn = gr.Button("👎 Bad")
    feedback_result = gr.Textbox(label="Feedback", interactive=False)

    submit_btn.click(fn=say_hi, inputs=name_input, outputs=output)
    good_btn.click(fn=handle_feedback, inputs=[output, gr.Textbox(value="Good", visible=False)], outputs=feedback_result)
    bad_btn.click(fn=handle_feedback, inputs=[output, gr.Textbox(value="Bad", visible=False)], outputs=feedback_result)

demo.launch()