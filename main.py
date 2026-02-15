from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/models")
async def get_models():
    return

@app.get("/models/ACE-Step/Ace-Step1.5")
async def get_model():
    model = AutoModel.from_pretrained("ACE-Step/Ace-Step1.5", trust_remote_code=True, dtype="auto")
    return model



@app.get("/models/${model_name}")
async def get_model(model_name: str):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype="auto")
    return model


