# Multimodal Misinformation Analyzer 🔍📸

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)

An advanced Multimodal AI API built to detect fake news and misinformation by analyzing both textual content and visual context (images) simultaneously. This module serves as a core component of the **TrueScope** fact-checking system.

## 🚀 Features
* **Multimodal Synergy:** Combines visual context (images) and textual claims to detect contradictions and sophisticated fake news.
* **Fine-Tuned LLM:** Utilizes a custom fine-tuned `gemma-3-4b-it` base model with LoRA adapters (PEFT) trained on rumor and fake news datasets.
* **Production-Ready API:** High-performance, asynchronous REST API built with FastAPI.
* **Containerized Deployment:** Fully containerized using Docker for seamless deployment on cloud platforms like Hugging Face Spaces.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Hugging Face Transformers, PEFT (LoRA), Safetensors
* **Backend:** FastAPI, Uvicorn, Requests
* **Image Processing:** Pillow (PIL)
* **Deployment:** Docker

## 📡 API Documentation

### Endpoint: `/predict`
Analyzes a given image URL and news text to classify the content as `real` or `fake`.

* **Method:** `POST`
* **Content-Type:** `application/json`

#### Request Body Example:
```json
{
  "image": "[https://upload.wikimedia.org/wikipedia/commons/4/46/Leonardo_Dicaprio_Cannes_2019.jpg](https://upload.wikimedia.org/wikipedia/commons/4/46/Leonardo_Dicaprio_Cannes_2019.jpg)",
  "text": "Is this news REAL or FAKE? Content: Breaking: Leonardo DiCaprio has officially announced that he is quitting acting forever to become a full-time monk in Tibet.\nAnswer:"
}
```

#### Response Example:
```json
{
  "verdict": "fake",
  "confidenceScore": 98.24
}
```
*Note: The `confidenceScore` is mathematically derived from the model's Softmax probabilities for the generated token.*


   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
