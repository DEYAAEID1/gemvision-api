# 💎 GemVision API

FastAPI-powered backend for gemstone image classification using a pretrained ViT model from Hugging Face.

🎯 This API receives an image of a gemstone and returns the predicted label using computer vision.

---

## 🚀 How It Works

The backend uses:

- `google/vit-base-patch16-224` Vision Transformer (Hugging Face)
- `FastAPI` for serving the API
- `Uvicorn` as the ASGI server

---

## 🧪 Example Usage

Send a POST request with a JPG/PNG image:

