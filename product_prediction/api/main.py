from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .inference import predict

app = FastAPI(title="Fashion Product Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Fashion Classifier API is running"}

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predictions = predict(image_file=image_bytes)
    return predictions
