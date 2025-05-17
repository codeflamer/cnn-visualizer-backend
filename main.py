from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import util as util
import time

util.load_artifacts()

app = FastAPI(title="Neural Network Visualization Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Neural Network Visualization Backend"}

@app.get("/all_layers")
async def get_all_layer():
    all_layers = util.get_all_layers() 
    return {"response": all_layers}

@app.get("/layer/{idx}")
async def get_layer(idx:int):
    layers = util.get_layer_tensor(idx)  # Convert integer to string
    return {"response": layers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 