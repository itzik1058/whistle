import ffmpeg
from numpy import float32, frombuffer
from numpy.typing import NDArray


def transform_audio(audio: bytes) -> NDArray[float32]:
    stdout, _ = (
        ffmpeg.input("pipe:", threads=0)
        .output("-", format="f32le", ac=1, ar=16000)
        .run(input=audio, capture_stdout=True, capture_stderr=True)
    )
    return frombuffer(stdout, float32)
