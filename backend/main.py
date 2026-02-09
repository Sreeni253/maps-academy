from fastapi import FastAPI

# This is the "Engine" of your website
app = FastAPI()

# This is the first message someone sees when they visit the API
@app.get("/")
def home():
    return {"message": "MAPS Academy Backend is Running"}

# This is a sample 'Data Point' for a map
@app.get("/test-map")
def test_map():
    return {
        "location": "Academy Center",
        "lat": 17.3850,
        "lng": 78.4867
    }
