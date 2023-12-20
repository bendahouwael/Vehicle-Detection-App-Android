# Vehicle Detection App using OpenCV and TensorFlow Lite

This Android application is designed for real-time vehicle detection using OpenCV and TensorFlow Lite. The project utilizes a pre-trained deep learning model (SSD MobileNet) to detect vehicles in live camera feed. The app is developed in Java and leverages the power of OpenCV for image processing and TensorFlow Lite for efficient on-device machine learning inference.

## Getting Started

### Prerequisites

- Android Studio
- OpenCV for Android
- TensorFlow Lite

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vehicle-detection-app.git
   
2.Open the project in Android Studio.

3.Ensure that OpenCV and TensorFlow Lite dependencies are correctly configured.

4.Build and run the application on an Android device.

## Usage

- Launch the application on your Android device.
- Grant camera permissions when prompted.
- Point the camera towards a scene with vehicles.
- Observe real-time vehicle detection with bounding boxes and labels.

## Project Structure

- **MainActivity**: The main activity that serves as the entry point of the application. It initializes OpenCV and handles the navigation to the CameraActivity.

- **CameraActivity**: The activity responsible for camera integration, frame processing, and displaying the real-time vehicle detection results. It utilizes the `objectDetectorClass` for inference.

- **objectDetectorClass**: The class that encapsulates the TensorFlow Lite model loading, image preprocessing, and vehicle detection logic.

## Model and Labels

- The TensorFlow Lite model (`ssd_mobilenet.tflite`) is used for vehicle detection.

- The labels used for classifying the detected vehicles are loaded from the `labelmap.txt` file.

## Contributing

Feel free to contribute to the project by opening issues or creating pull requests. Contributions are always welcome!

## Acknowledgments

Special thanks to the contributors of OpenCV and TensorFlow Lite for their excellent libraries.
