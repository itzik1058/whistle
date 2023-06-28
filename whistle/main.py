from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from librosa import resample
from numpy import array, float32, int16
from pydub import AudioSegment
from whispercpp import Whisper

model: Whisper | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = Whisper.from_pretrained("tiny.en")
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
            print("received", len(data))
            try:
                segment: AudioSegment = AudioSegment.from_file(BytesIO(data))
                audio = (
                    array(
                        segment.get_array_of_samples(),
                        dtype=int16,
                    ).astype(float32)
                    / 32768
                )
                audio = resample(audio, orig_sr=segment.frame_rate, target_sr=16_000)
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
