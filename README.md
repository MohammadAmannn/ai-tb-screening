# AI-Based Tuberculosis Screening Tool

## Problem
Tuberculosis diagnosis from chest X-rays requires trained radiologists,
who are scarce in many regions. Delayed diagnosis increases disease
spread and mortality.

## Solution
An AI-assisted screening system using a Convolutional Neural Network
(CNN) to analyze chest X-ray images and flag potential TB risk, helping
healthcare workers prioritize patients for confirmatory testing.

## Tech Stack
- Python
- TensorFlow / Keras
- CNN (ResNet50 with Transfer Learning)
- OpenCV, PIL
- Streamlit

## Project Structure
- training/ → Model training notebook (Colab/Kaggle)
- app/ → Inference and web demo

## How to Run
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
