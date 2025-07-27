# 🤗 HuggingFace 
📌 Project Overview
This project demonstrates the use of Hugging Face Transformers for various Natural Language Processing (NLP) and Generative AI tasks.

✅ Text Generation using GPT-2

✅ Named Entity Recognition (NER)

✅ Sentiment Analysis

✅ Text-to-Image generation with Stable Diffusion

✅ Interactive UI with Gradio

The entire workflow is implemented in Google Colab with GPU acceleration (Tesla T4).

🛠️ Tech Stack
Python 3

Hugging Face Transformers

Diffusers (Stable Diffusion)

Gradio (for UI)

PyTorch (GPU support)

📂 Project Modules
1️⃣ Text Generation with GPT-2

Uses Hugging Face pipeline("text-generation")

Generates coherent text based on a prompt

from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")

generator("What is Hugging Face?", max_length=50, num_return_sequences=1)

2️⃣ Named Entity Recognition (NER)

Uses BERT-based NER model

Identifies PERSON, ORG, LOC entities in text

ner = pipeline("ner", grouped_entities=True)

ner("Hugging Face is based in NYC and partners with Google")

3️⃣ Sentiment Analysis

Uses DistilBERT fine-tuned on SST-2

Classifies text as POSITIVE or NEGATIVE

classifier = pipeline("sentiment-analysis")

classifier("I love Hugging Face Library very much!")

✅ Positive example → POSITIVE

✅ Negative example → NEGATIVE

4️⃣ Text-to-Image with Stable Diffusion

Uses Stable Diffusion v1.5 for AI-generated images

Generates images from natural language prompts

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5")

pipe.to("cuda")

image = pipe("A photo of a peacock feather").images[0]

image.save("peacock_feather.png")

5️⃣ Gradio Web UI

Interactive UI for Sentiment Analysis

Interactive UI for Text-to-Image Generation

import gradio as gr

gr.Interface(fn=analyze_sentiment, inputs="text", outputs="text").launch()

🚀 How to Run

2️⃣ Install Dependencies

pip install transformers diffusers gradio torch

3️⃣ Run in Google Colab with GPU

Ensure Tesla T4 GPU runtime

Open HF_Bootcamp.ipynb in Colab

4️⃣ Launch Gradio Apps

gr.Interface(fn=analyze_sentiment, inputs="text", outputs="text").launch()

📊 Features

✅ Text Generation → Generate creative text with GPT-2

✅ NER → Identify named entities (ORG, PER, LOC)

✅ Sentiment Analysis → Positive/Negative classification

✅ AI Image Generation → Convert text prompts into images

✅ Web UI → Easy interaction using Gradio

📌 Sample Outputs
Text Generation:

“What is Hugging Face? Hugging Face is an amazing AI platform that allows you…”

NER:
[{'entity_group': 'ORG', 'word': 'Hugging Face', 'score': 0.95}]

Sentiment Analysis:



[{'label': 'POSITIVE', 'score': 0.99}]

Text-to-Image:

🖼️ AI-generated peacock feather image

🔮 Future Enhancements

Add Translation & Summarization models

Deploy on Hugging Face Spaces

Integrate Voice-to-Text & Chatbot UI
