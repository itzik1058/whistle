from contextlib import asynccontextmanager

import ffmpeg
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from numpy import float32, frombuffer
from whispercpp import Whisper

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
            try:
                stdout, _ = (
                    ffmpeg.input("pipe:", threads=0)
                    .output("-", format="f32le", ac=1, ar=16000)
                    .run(input=data, capture_stdout=True, capture_stderr=True)
                )
                audio = frombuffer(stdout, float32)
                transcript = model.transcribe(audio)
                print(transcript)
                await websocket.send_text(transcript)
            except KeyboardInterrupt:
                await websocket.close()
                return
            except Exception as e:
                print(e)
    except WebSocketDisconnect:
        pass
