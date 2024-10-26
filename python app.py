from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO  # Import YOLO from ultralytics

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO('best.pt')  # Replace with the path to your trained model

# Define plant disease information (same as your provided dictionary)
plant_disease_info = {
    'Apple Scab Leaf': {
        'precaution': 'Remove infected leaves and improve air circulation.',
        'treatment': 'Apply fungicides like myclobutanil or captan to control the spread.'
    },
    'Apple leaf': {
        'precaution': 'Monitor regularly for any changes.',
        'treatment': 'No treatment needed if healthy.'
    },
    'Apple rust leaf': {
        'precaution': 'Prune infected areas and avoid overhead watering.',
        'treatment': 'Use fungicides like chlorothalonil or sulfur sprays.'
    },
    'Bell_pepper leaf spot': {
        'precaution': 'Avoid wetting foliage and improve air circulation.',
        'treatment': 'Apply a copper-based fungicide or use resistant varieties.'
    },
    'Bell_pepper leaf': {
        'precaution': 'Monitor for signs of disease.',
        'treatment': 'No treatment needed if healthy.'
    },
    # Add other classes similarly...
    'Tomato mold leaf': {
        'precaution': 'Avoid excessive watering and improve air flow.',
        'treatment': 'Apply sulfur dust or a copper-based fungicide.'
    },
    'Tomato two spotted spider mites leaf': {
        'precaution': 'Introduce natural predators or use insecticidal soap.',
        'treatment': 'Spray with neem oil or horticultural oil to control the infestation.'
    },
    'grape leaf black rot': {
        'precaution': 'Prune infected vines and ensure good air circulation.',
        'treatment': 'Apply fungicides like captan or mancozeb to prevent spread.'
    },
    'grape leaf': {
        'precaution': 'Monitor regularly for any changes.',
        'treatment': 'No treatment needed if healthy.'
    },
    # Add the rest of the disease classes as needed...
}

# Initialize webcam
# Initialize webcam
ip_camera_url = "http://192.168.1.5:8080/video"  # Replace with your IP camera URL
camera = cv2.VideoCapture(ip_camera_url)

# Set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set desired width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set desired height


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run YOLO inference
            results = model.predict(frame, imgsz=640, conf=0.25)  # Adjust confidence threshold if needed

            # Loop over each detection
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
                confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Get class labels

                # Draw bounding boxes and display disease information
                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers

                    # Check if class_id is valid
                    class_id_int = int(class_id)
                    if class_id_int < 0 or class_id_int >= len(plant_disease_info):
                        disease_name = "Unknown Disease"
                    else:
                        disease_name = list(plant_disease_info.keys())[class_id_int]  # Class name

                    label = f"{disease_name} ({confidence:.2f})"

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display disease name
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Get precautions and treatments if disease is known
                    if disease_name in plant_disease_info:
                        precautions = plant_disease_info[disease_name]['precaution']
                        treatment = plant_disease_info[disease_name]['treatment']

                        # Display precautions and treatment info on the frame
                        cv2.putText(frame, f"Precaution: {precautions}", (20, frame.shape[0] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, f"Treatment: {treatment}", (20, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
