import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from whispercpp import Whisper

from whistle.utils import transform_audio

model: Whisper | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = Whisper.from_pretrained("tiny")
    yield
    model = None


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if model is None:
        await websocket.send_text("model is not loaded")
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_bytes()
            loop = asyncio.get_running_loop()
            try:
                audio = await loop.run_in_executor(None, transform_audio, data)
                transcript = await loop.run_in_executor(None, model.transcribe, audio)
                print(transcript)
                await websocket.send_text(transcript)
            except Exception as e:
                print(e)
    except WebSocketDisconnect:
        pass
