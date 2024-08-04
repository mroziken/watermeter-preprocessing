from flask import Flask
import paho.mqtt.client as mqtt
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
import io
import logging
import numpy as np
from scipy.ndimage import label, find_objects
from skimage.metrics import structural_similarity as ssim
import os
import json
import dotenv


# Load environment variables from .env file
dotenv.load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

def maskSmallObjects(image):
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
    last_file_path = '/home/pi/watermeter/preprocessed/last_file_name.jpeg'

    try:
        image = Image.open(file_path)

        if os.path.exists(last_file_path):
            last_image = Image.open(last_file_path)

            if compare_images(image, last_image):
                logging.info('New image is significantly different from the last processed image, proceeding with transformation.')
                image = transform_image(image)

                output_path = os.path.join('/home/pi/watermeter/preprocessed', os.path.basename(file_path))
                image.save(output_path, format='JPEG')
                logging.info(f"Processed image saved to {output_path}")
            else:
                logging.info('New image is similar to the last processed image, skipping transformation.')
        else:
            logging.info('No last image found, proceeding with transformation.')
            image = transform_image(image)

            output_path = os.path.join('/home/pi/watermeter/preprocessed', os.path.basename(file_path))
            image.save(output_path, format='JPEG')
            logging.info(f"Processed image saved to {output_path}")

        # Save the new image as the last processed image
        image.save(last_file_path, format='JPEG')

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
