from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Crop images (make sure all these images are in static/ folder)
CROP_IMAGE_MAP = {
    "rice": "rice.jpg",
    "wheat": "wheat.jpeg",
    "millet": "millet.jpeg",
    "maize": "maize.jpg",
    "chickpea": "chickpea.jpg",
    "lentil": "lentil.jpg",
    "pigeonpea": "pigeonpea.jpeg",
    "groundnut": "groundnut.jpg",
    "barley": "barley.jpeg"
}

# Crop descriptions
# Crop descriptions with care tips
DESCRIPTIONS = {
    "rice": "Rice grows well in high humidity, high rainfall, and clayey soil. It needs standing water and nitrogen fertilizer for good yield.",
    "wheat": "Wheat prefers moderate temperature and well-drained loamy soil. It requires regular irrigation and phosphorus-rich fertilizer.",
    "maize": "Maize grows well in moderate rainfall and fertile soil. It needs plenty of sunlight, potash, and nitrogen for healthy growth.",
    "chickpea": "Chickpea thrives in low rainfall and sandy soil. It benefits from organic manure and requires less irrigation.",
    "pigeonpea": "Pigeonpea tolerates drought and grows in warm weather. It needs deep soil and benefits from organic compost.",
    "lentil": "Lentil prefers cooler climate and well-drained soil. It needs moderate watering and grows well with phosphate fertilizers.",
    "millet": "Millet is drought-resistant and grows in less fertile soil. It requires very little water and benefits from organic manure.",
    "barley": "Barley grows in cool climate and sandy loam soil. It requires moderate irrigation and nitrogen fertilizer.",
    "groundnut": "Groundnut prefers sandy soil with good drainage. It needs gypsum and calcium, with proper watering at the flowering stage."
}


# Alternatives (only 1 alternative now)
ALTERNATIVES = {
    "rice": "wheat",
    "wheat": "barley",
    "maize": "millet",
    "chickpea": "lentil",
    "pigeonpea": "chickpea",
    "lentil": "wheat",
    "millet": "maize",
    "barley": "wheat",
    "groundnut": "maize"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        data = [float(x) for x in request.form.values()]
        final_input = [np.array(data)]
        prediction = model.predict(final_input)[0]

        key = prediction.lower().replace(" ", "")
        image_file = CROP_IMAGE_MAP.get(key, None)
        description = DESCRIPTIONS.get(key, "This crop matches your soil and climate conditions.")
        alternative = ALTERNATIVES.get(key, None)

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
    app.run(debug=True, use_reloader=False)
