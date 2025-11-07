from fastapi import FastAPI, Form
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np
import requests
from io import BytesIO
import base64

app = FastAPI(
    title="Canny Edge API",
    description="API para gerar mapas de bordas (Canny Edge) a partir de URLs de imagem.",
    version="1.0"
)

@app.post("/canny")
async def generate_canny(
    image_url: str = Form(...),
    low_threshold: int = Form(100),
    high_threshold: int = Form(200)
):
    """
    Recebe uma URL de imagem e retorna:
    - A imagem processada (Canny Edge)
    - A imagem em Base64 no JSON
    """

    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return JSONResponse(status_code=400, content={"error": "Erro ao baixar a imagem."})

        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        success, buffer = cv2.imencode(".png", edges)
        if not success:
            return JSONResponse(status_code=500, content={"error": "Falha ao gerar imagem Canny."})

        image_base64 = base64.b64encode(buffer).decode("utf-8")

        headers = {"Content-Type": "application/json"}
        return JSONResponse(
            content={
                "status": "success",
                "image_url": image_url,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "image_base64": image_base64
            },
            headers=headers
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})