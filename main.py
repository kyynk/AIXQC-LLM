from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Templates configuration
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_text(request: Request, text: str = None):
    processed_text = f"You entered: {text}"
    return templates.TemplateResponse("index.html", {"request": request, "processed_text": processed_text})
