from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection
import os
import tensorflow as tf
import cv2


execution_path = os.getcwd()


def train_detection_model():
    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="fire-dataset")
    trainer.setTrainConfig(object_names_array=["fire"], batch_size=8, num_experiments=100,
                           train_from_pretrained_model="pretrained-yolov3.h5")
    trainer.trainModel()


def detect_from_image():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "2.jpg"),
                                                 output_image_path=os.path.join(execution_path, "2-detected.jpg"),
                                                 minimum_percentage_probability=40)

    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_from_video():
    detector = CustomVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()
    camera = cv2.VideoCapture(0)
    Frame=80;
    fps=Frame/10;
    dt=Frame/fps;
    detected_video_path = detector.detectObjectsFromVideo(camera_input=camera, frames_per_second=fps, output_file_path=os.path.join(execution_path, "video_Cam-detected"), minimum_percentage_probability=20, log_progress=True, detection_timeout=dt )
    print("Timeout");


if __name__ == '__main__':
    #train_detection_model();
    #detect_from_image();
    detect_from_video();
