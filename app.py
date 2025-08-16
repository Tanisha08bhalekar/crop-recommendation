from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Folder where images are stored
STATIC_FOLDER = "static"

# All crop keys
ALL_CROPS = [
    "rice", "wheat", "millet", "maize", "chickpea",
    "lentil", "pigeonpea", "groundnut", "barley",
    "mothbeans", "kidneybeans"
]

# Automatically detect images in static/ folder
CROP_IMAGE_MAP = {}
for crop in ALL_CROPS:
    matched_files = [f for f in os.listdir(STATIC_FOLDER) if crop.lower() in f.lower()]
    if matched_files:
        CROP_IMAGE_MAP[crop] = matched_files[0]  # use first match
    else:
        CROP_IMAGE_MAP[crop] = "placeholder.jpg"  # ensure placeholder exists
        print(f"Warning: No image found for {crop}, using placeholder.")

print("CROP_IMAGE_MAP:", CROP_IMAGE_MAP)


# Crop descriptions
DESCRIPTIONS = {
    "rice": "Rice grows well in high humidity, high rainfall, and clayey soil. It needs standing water and nitrogen fertilizer for good yield.",
    "wheat": "Wheat prefers moderate temperature and well-drained loamy soil. It requires regular irrigation and phosphorus-rich fertilizer.",
    "maize": "Maize grows well in moderate rainfall and fertile soil. It needs plenty of sunlight, potash, and nitrogen for healthy growth.",
    "chickpea": "Chickpea thrives in low rainfall and sandy soil. It benefits from organic manure and requires less irrigation.",
    "pigeonpea": "Pigeonpea tolerates drought and grows in warm weather. It needs deep soil and benefits from organic compost.",
    "lentil": "Lentil prefers cooler climate and well-drained soil. It needs moderate watering and grows well with phosphate fertilizers.",
    "millet": "Millet is drought-resistant and grows in less fertile soil. It requires very little water and benefits from organic manure.",
    "barley": "Barley grows in cool climate and sandy loam soil. It requires moderate irrigation and nitrogen fertilizer.",
    "groundnut": "Groundnut prefers sandy soil with good drainage. It needs gypsum and calcium, with proper watering at the flowering stage.",
    "mothbeans": "Moth beans grow in arid and semi-arid regions with well-drained soil. They require minimal irrigation and are rich in protein.",
    "kidneybeans": "Kidney beans grow in warm climates with loamy soil. They require regular irrigation and moderate nitrogen fertilizer."
}

# Alternatives
ALTERNATIVES = {
    "rice": "wheat",
    "wheat": "barley",
    "maize": "millet",
    "chickpea": "lentil",
    "pigeonpea": "chickpea",
    "lentil": "wheat",
    "millet": "maize",
    "barley": "wheat",
    "groundnut": "maize",
    "mothbeans": "kidneybeans",
    "kidneybeans": "mothbeans"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = [np.array(data)]
        prediction = model.predict(final_input)[0]

        key = prediction.lower().replace(" ", "")
        image_file = CROP_IMAGE_MAP.get(key, "placeholder.jpg")
        description = DESCRIPTIONS.get(key, "This crop matches your soil and climate conditions.")
        alternative = ALTERNATIVES.get(key)

        return render_template(
            "result.html",
            prediction=prediction,
            image_file=image_file,
            description=description,
            alternative=alternative
        )
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
