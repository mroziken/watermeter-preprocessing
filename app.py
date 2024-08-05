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
import io


# Load environment variables from .env file
dotenv.load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CROP_PARAMS = [
    (530, 670, 574, 725), (574, 670, 618, 725), (618, 670, 662, 725), (662, 670, 706, 725),
    (706, 670, 750, 725), (750, 670, 794, 725), (794, 670, 838, 725), (838, 670, 882, 725)
]

app = Flask(__name__)

def crop_and_resize(image,file_name):
    logging.info('In crop_and_resize')
    cropped_images = []
    for crop_param in CROP_PARAMS:
        cropped_image = image.crop(crop_param)
        cropped_image = cropped_image.resize((640, int((640 / cropped_image.width) * cropped_image.height)), Image.Resampling.BICUBIC)
        
        # cropped_file_name should be file_name-0.jpeg, file_name-1.jpeg, etc.
        cropped_file_name = f"{file_name}-{len(cropped_images)}"

        cropped_images.append((cropped_image, cropped_file_name))

    return cropped_images

def predict_image_classification(image, file_name):
    logging.info('In predit_image_classification')
    
    # image variable is a PIL Image object
    # Send image to endpoint at http://192.168.1.22:5000/predict
    # Use the requests library to send a POST request with the image
    # The response will be a JSON object with the classification results
    # Example response: {"file_name":"0033baa78cc45f7a-1.jpeg","prediction":0}
    # Return the prediction from the response
    
    url = "http://192.168.1.22:5000/predict"
    files = {'image': image}
    response = requests.post(url, files=files)
    prediction = response.json()["prediction"]

    # Save the image to the /home/pi/watermeter/cropped directory as file_name-prediction.jpeg
    cropped_path = f"/home/pi/watermeter/cropped/{file_name}-{prediction}.jpeg"
    image.save(cropped_path)

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

        output_path = os.path.join('/home/pi/watermeter/preprocessed', os.path.basename(file_path))
        image.save(output_path, format='JPEG')
        logging.info(f"Processed image saved to {output_path}")
        cropped_images = crop_and_resize(image,file_name)

        # Iterate through cropped_images, convert images to bytes-like objects
        # Save the results in cropped_images_bytes list as tuples (bytes-like object, cropped_file_name string)
        cropped_images_bytes = [(io.BytesIO(), cropped_file_name) for _, cropped_file_name in cropped_images]
        final_cropped_images_bytes = []
        for bytes_io, cropped_file_name in cropped_images_bytes:
            cropped_image = next(cropped_image for cropped_image, name in cropped_images if name == cropped_file_name)
            cropped_image.save(bytes_io, format='JPEG')
            bytes_io.seek(0)
            final_cropped_images_bytes.append((bytes_io, cropped_file_name))

        # Iterate through final_cropped_images_bytes, call predict_image_classification for each image
        # Save the results in classifications list
        classifications = [predict_image_classification(cropped_image, cropped_file_name) for cropped_image, cropped_file_name in final_cropped_images_bytes]

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
        if os.path.exists(file_path):
            process_image(file_path)
        else:
            logging.error(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
    except KeyError as e:
        logging.error(f"Missing key in JSON data: {e}")

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
