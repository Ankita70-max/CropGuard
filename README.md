# CropGuard-AI
AI-Based Crop Disease Detection and Treatment Recommendation System.

## Run
Python 3.10 recommended.
1. `py -3.10 -m venv venv`
2. `venv\\Scripts\\activate`
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open http://127.0.0.1:5000

The app runs immediately in demo mode. Add a trained model at `model/cropguard_model.h5` and `model/class_names.json` to enable real CNN predictions.
