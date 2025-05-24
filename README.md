---
title: Promptify
emoji: 👁️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
---

# Promptify

Promptify is an AI-powered app that converts human task demonstrations into automation-ready descriptions.

Upload a short screen recording or task video, and Promptify will:
1. Extract a representative frame
2. Use a vision-language AI model to describe the visual task in plain language

## What It Does

- ✅ Accepts video input (screen recording or webcam)
- ✅ Extracts a frame using OpenCV
- ✅ Runs the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) image captioning model
- ✅ Outputs a natural-language description of the task

## Tech Stack

- [Gradio](https://gradio.app) + [Hugging Face Spaces](https://huggingface.co/spaces)  
- `opencv-python-headless` – video frame extraction  
- `transformers` + `torch` – to load and run BLIP model  
- Python – simple backend, no frontend code

## Try the Live Demo

👉 [Launch Promptify on Hugging Face](https://huggingface.co/spaces/zephyr-io/promptify)

## Files

- `app.py` – Main application logic
- `requirements.txt` – Dependencies for Hugging Face Space
- `.huggingface.yaml` – Hugging Face config
- `README.md` – This file

## Credits

Built with ❤️ by [@YTibs](https://github.com/YTibs)  
[Hugging Face](https://huggingface.co) and [Gradio](https://gradio.app)