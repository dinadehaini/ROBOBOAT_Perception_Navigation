import os
from ultralytics import YOLO  # Import YOLO from ultralytics
from roboflow import Roboflow
# Your RoboFlow API key
API_KEY = 'zrxPxxPkGjehXaCiZShF'

# Dataset link from RoboFlow
DATASET_URL = f"https://universe.roboflow.com/api/datasets/3/export?api_key={API_KEY}&name=cornell-autoboat-fz9yv&export-format=yolov8"

# Path to store the dataset
DATASET_PATH = './data'

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
# model.predict(
#    source='https://media.roboflow.com/notebooks/examples/dog.jpeg',
#    conf=0.25
# )

# rf = Roboflow(api_key="zrxPxxPkGjehXaCiZShF")
# project = rf.workspace().project("buoy-detection-dzz7y")
# model = project.version("1").model

# yolo task=detect \
# mode=train \
# model=yolov8s.pt \
# data={dataset.location}/data.yaml \
# epochs=100 \
# imgsz=640



def download_dataset(url, path):
    os.makedirs(path, exist_ok=True)
    command = f"curl -L \"{url}\" | tar -xz -C {path}"
    os.system(command)

# Function to train the model
def train_yolov8():
    data_path = "/Users/dinadehaini/Downloads/Official Buoy Detection.v4i.yolov8-obb/data.yaml"
    train_command = f'yolo train data="{data_path}" model=yolov8n.pt epochs=50 batch=16 imgsz=640'
    os.system(train_command)

def main():
    print("Downloading the dataset...")
    download_dataset(DATASET_URL, DATASET_PATH)
    
    print("Starting training...")
    train_yolov8()

if __name__ == "__main__":
    main()