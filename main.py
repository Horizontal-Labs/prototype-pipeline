from fastapi import FastAPI
from dotenv import load_dotenv
from app.routes import router

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Argument Mining API",
    description="API for analyzing argumentative structures in text",
    version="0.2.0"
)

# Include routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)