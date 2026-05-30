from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
import torch
import requests
import io
import uvicorn
import os

app = FastAPI()

class PredictionInput(BaseModel):
    image: str
    text: str

hf_token = os.getenv("HF_TOKEN")
base_model_name = "google/gemma-3-4b-it"
lora_path = r"D:\multi_copy\checkpoint-408"

processor = AutoProcessor.from_pretrained(
    base_model_name,
    token=hf_token
)

base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map={"":"cpu"},
    token=hf_token
)

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

@app.post("/predict")
async def predict(item: PredictionInput):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(item.image, headers=headers, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Cannot download image")
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item.text}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
            
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        decoded_output = processor.decode(generated_tokens, skip_special_tokens=True).lower()
        
        confidence_score = 0.0
        if len(outputs.scores) > 0:
            first_token_logits = outputs.scores[0]
            first_token_probs = torch.nn.functional.softmax(first_token_logits, dim=-1)
            first_token_id = generated_tokens[0]
            confidence_score = round(first_token_probs[0, first_token_id].item() * 100, 2)
            
        verdict = "fake" if "fake" in decoded_output else "real"
        
        return {
            "verdict": verdict,
            "confidenceScore": confidence_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)