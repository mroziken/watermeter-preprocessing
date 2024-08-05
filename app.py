from flask import Flask
import paho.mqtt.client as mqtt
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
import logging
import numpy as np
from scipy.ndimage import label, find_objects
from skimage.metrics import structural_similarity as ssim
import os
import json
import dotenv
from datetime import datetime
import requests
from io import BytesIO

# Load environment variables from .env file
dotenv.load_dotenv()

PREDICTION_API_ENDPOINT = os.getenv("PREDICTION_API_ENDPOINT", "localhost:5000")
WATERMETER_BASE_DIR = os.getenv("WATERMETER_BASE_DIR", "/home/pi/watermeter")
IN = f'{WATERMETER_BASE_DIR}/in'
PREPROCESSED = f'{WATERMETER_BASE_DIR}/preprocessed'
CROPPED = f'{WATERMETER_BASE_DIR}/cropped'
PREDICTIONS = f'{WATERMETER_BASE_DIR}/predictions'
PROCESSED = f'{WATERMETER_BASE_DIR}/processed'
CROP_PARAMS = [
    (530, 670, 574, 725), (574, 670, 618, 725), (618, 670, 662, 725), (662, 670, 706, 725),
    (706, 670, 750, 725), (750, 670, 794, 725), (794, 670, 838, 725), (838, 670, 882, 725)
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)

# Set to keep track of processed filenames
processed_files = set()

def crop_and_resize(file_name):
    logging.info('In crop_and_resize')
    base_file_name = os.path.basename(file_name)
    cropped_images = []
    image = Image.open(file_name)
    for crop_param in CROP_PARAMS:
        cropped_image = image.crop(crop_param)
        cropped_image = cropped_image.resize((640, int((640 / cropped_image.width) * cropped_image.height)), Image.Resampling.BICUBIC)
        
        # cropped_file_name should be file_name-0.jpeg, file_name-1.jpeg, etc.
        cropped_file_name = f"{CROPPED}/{base_file_name}-{len(cropped_images)}.jpeg"
        cropped_image.save(cropped_file_name)

        cropped_images.append(cropped_file_name)

    return cropped_images


def predict_image_classification(file_name):
    logging.info('In predict_image_classification')
    url = PREDICTION_API_ENDPOINT

    # Extract the base file name from the file_name
    base_file_name = os.path.basename(file_name)
    
    # Open the image file
    image = Image.open(file_name)

    # Convert the image to a bytes-like object
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    # Prepare the file for the POST request
    files = {'file': (base_file_name, image_bytes, 'image/jpeg')}
    logging.info(f"Sending image to prediction API: {url}")
    
    # Send the POST request
    response = requests.post(url, files=files)
    response.raise_for_status()  # Raise an error for bad responses

    prediction = response.json()
    logging.info(f"Prediction: {prediction}")

    # Save the prediction in a file
    if not os.path.exists(CROPPED):
        os.makedirs(CROPPED)
    cropped_with_prediction = f"{CROPPED}/{base_file_name}-{prediction}.jpeg"
    image.save(cropped_with_prediction)

    return prediction


def extract_label(prediction):
    logging.info('In extract_label')
    try:
        display_names = prediction["displayNames"]
        if display_names:
            return display_names[0]
    except KeyError as e:
        print(f"KeyError: {e}")
    except AttributeError as e:
        print(f"AttributeError: {e}")
    return 'Unknown'


def maskSmallObjects(image):
    logging.info('In maskSmallObjects')
    # Convert to grayscale
    gray = image.convert('L')
    
    # Apply Gaussian Blur
    blurred = gray.filter(ImageFilter.GaussianBlur(2))
    
    # Apply thresholding
    thresholded = blurred.point(lambda p: p > 128 and 255)
    
    # Convert image to numpy array
    np_image = np.array(thresholded)
    
    # Find contours using scipy.ndimage
    labeled_array, num_features = label(np_image == 0)
    objects = find_objects(labeled_array)
    
    # Create a mask for the numbers
    mask = Image.new('L', gray.size, 0)
    draw = ImageDraw.Draw(mask)
    
    for obj in objects:
        x1, y1, x2, y2 = obj[1].start, obj[0].start, obj[1].stop, obj[0].stop
        if (x2 - x1) > 10 and (y2 - y1) > 10:  # Filter out small objects
            draw.rectangle([x1, y1, x2, y2], fill=255)
    
    # Bitwise-and to extract the numbers from the original image
    result = Image.composite(gray, Image.new('L', gray.size, 255), mask)
    
    return result

def transform_image(image):
    logging.info('In transform_image')
    # Image processing steps
    # Rotate the image by 7.5 degrees
    image = image.rotate(7.5, expand=False)
    image = ImageOps.autocontrast(image)

    image_cropped = image.crop((530, 670, 880, 740))  # Crop
    image_cropped = image_cropped.rotate(180, expand=False)

    image_cropped = maskSmallObjects(image_cropped)

    image = Image.new('RGB', (1600, 800))
    image.paste(image_cropped,(530, 670))
        
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)

def compare_images(img1, img2, threshold=0.99):
    logging
    # Convert images to grayscale
    img1_gray = img1.convert('L')
    img2_gray = img2.convert('L')

    # Convert images to numpy arrays
    img1_array = np.array(img1_gray)
    img2_array = np.array(img2_gray)

    # Compute SSIM between the two images
    similarity, _ = ssim(img1_array, img2_array, full=True)
    logging.info(f"threshold: {threshold}, similarity: {similarity}")
    # If similarity is below the threshold, images are considered different
    return similarity < threshold

def process_image(file_path):
    logging.info(f"Processing image: {file_path}")

    try:
        image = Image.open(file_path)
        file_name = os.path.basename(file_path)


        logging.info('No last image found, proceeding with transformation.')
        image = transform_image(image)

        preprocessed_file = os.path.join(PREPROCESSED, os.path.basename(file_path))
        image.save(preprocessed_file, format='JPEG')
        logging.info(f"Processed image saved to {preprocessed_file}")
        cropped_images = crop_and_resize(preprocessed_file)

 

        # Iterate through cropped_images, call predict_image_classification for each image
        # Save the results in classifications list


        classifications = [predict_image_classification(cropped_image) for cropped_image in cropped_images]

        classification_labels = [extract_label(prediction) for classification in classifications for prediction in classification]


        meter_readings = ""
        for i, label in enumerate(classification_labels):
            meter_readings += label

        # Prepare the message to be published
        message = json.dumps({
            'imageName': file_name,
            'value': meter_readings,
            'timestamp': datetime.now().isoformat(),
            'version': 3.0
        })

        # Publish the message to the MQTT topic watermeter-out
        client.publish("watermeter-out", message)
        logging.info(f"Published message to watermeter-out: {message}")

    except Exception as error:
        logging.error(f'Failed to process image: {error}')

# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected with result code {rc}")
    client.subscribe("watermeter-in")

def on_message(client, userdata, msg):
    logging.info(f"Received message on {msg.topic}: {msg.payload.decode()}")
    try:
        data = json.loads(msg.payload.decode())
        file_path = data['filename']
        
        # Check if the file has already been processed
        if file_path not in processed_files:
            if os.path.exists(file_path):
                process_image(file_path)
                processed_files.add(file_path)  # Mark as processed
            else:
                logging.error(f"File not found: {file_path}")
        else:
            logging.info(f"File already processed: {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")

# Read MQTT configuration from environment variables
mqtt_broker_address = os.getenv("MQTT_BROKER_ADDRESS", "localhost")  # Default to localhost if not set
mqtt_broker_port = int(os.getenv("MQTT_BROKER_PORT", 1883))         # Default to 1883 if not set

# Start the MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(mqtt_broker_address, mqtt_broker_port, 60)  # Replace with your MQTT broker address

# Start a background thread for the MQTT loop
client.loop_start()

flask_app_port = os.getenv("FLASK_APP_PORT", 5000)
print(f'flask_app_port: {flask_app_port}')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010)
