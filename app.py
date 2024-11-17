from predict_leaf_area_index import LeafAreaIndexCalculator
import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

lf = LeafAreaIndexCalculator()
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded images
PROCESSED_FOLDER = 'output_results'  # Folder to store processed images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected file", 400

    try:
        # Secure the filename and save the file
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)

        # Get the absolute path
        full_path = os.path.abspath(file_path)
        print(full_path)

        lai = lf.predictLeafAreaIndex(full_path)

        # Process the image (e.g., resize to 200x200 pixels)
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"plot_image.png")
        processed_file_path = os.path.abspath(processed_path)

        print(f"Leaf area index : {lai}")

        # Return the URL of the processed image
        return jsonify({"processed_image_url": f"/output_results/{os.path.basename(processed_path)}", "leaf_area_index": f"{lai}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/output_results/<filename>')
def serve_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)