from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from inference import gradcam_inference  # Import your inference function
import os

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'static/uploads/'
EXAMPLE_FOLDER = 'static/examples/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to save and process uploaded image
def save_and_process_image(file):
    # Save the uploaded file to the uploads directory
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Run the inference function
    pred_class, cam_img_path = gradcam_inference(filepath)

    # Return the prediction and the paths to the images
    return pred_class, filepath, cam_img_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle example image selection
        example_image = request.form.get('example_image')
        if example_image:
            # Use selected example image
            example_image_path = os.path.join(EXAMPLE_FOLDER, example_image)
            pred_class, cam_img_path = gradcam_inference(example_image_path)
            return render_template(
                'index.html', 
                prediction=pred_class, 
                original_img_url=example_image_path, 
                cam_img_url=cam_img_path, 
                is_example=True
            )

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return redirect(request.url)

            if file:
                # Save and process the uploaded image
                pred_class, original_img_path, cam_img_path = save_and_process_image(file)

                return render_template(
                    'index.html', 
                    prediction=pred_class, 
                    original_img_url=original_img_path, 
                    cam_img_url=cam_img_path,
                    is_example=False
                )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
