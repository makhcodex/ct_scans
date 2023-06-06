from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


@app.get("/ping")
async def ping():
    return 'Hello'


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File()
):
    image = read_file_as_image(await file.read())
    image_pil = Image.fromarray(image).convert('RGB')
    image_rgb = np.array(image_pil)
    img_batch = np.expand_dims(image_rgb, axis=0)
    img_batch = img_batch.reshape(img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], 3)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.get("/")
async def qalaysiz():
    return{"message": "Please upload your image on '/predict' via POST request to obtain results"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

