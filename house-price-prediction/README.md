# House Price Prediction Service (Production-Ready)

This repository contains a high-performance ML microservice for predicting house prices. Instead of just a "script," we’ve built a **bridge** between a trained Linear Algebra model and a real-world application using **FastAPI** and **Docker**.

## The Philosophy
"If it only runs on your laptop, it’s not a product." 
This project follows the **Twelve-Factor App** methodology. We focus on **reproducibility** (it works everywhere) and **speed** (it handles multiple requests instantly).

## The Tech Stack (The "Why")

*   **FastAPI:** Chosen for its lightning-fast performance and automatic data validation (using Pydantic). It ensures that if a user sends a "string" instead of a "price," the API rejects it before it breaks the model.
*   **Docker:** We use a **multi-stage build** to keep the image "slim." This means faster deployments and less storage waste.
*   **Joblib:** Efficiently loads our pre-trained weights (the $W$ and $b$ matrices we studied in Linear Algebra).
*   **Gunicorn/Uvicorn:** A production-grade server setup that allows the API to handle concurrent users without crashing.

## How it Works (The Architecture)

1.  **Input:** The user sends house features (sq ft, bedrooms, age) via a POST request.
2.  **Validation:** FastAPI checks if the dimensions match our model's requirements.
3.  **Inference:** The model performs a quick matrix multiplication ($Y = WX + b$) behind the scenes.
4.  **Output:** The predicted price is returned in milliseconds.

## Quick Start (Get running in 2 minutes)

If you have **Docker** installed, just run:

```bash
# Build the "slim" image
docker build -t house-price-api .

# Run the container on port 8000
docker run -p 8000:8000 house-price-api

