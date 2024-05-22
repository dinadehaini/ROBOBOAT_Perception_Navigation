import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load the model
model = YOLO('/Users/dinadehaini/Downloads/best (2).pt')  # Ensure correct model path

# History queues to track past positions of buoys
red_buoy_history = deque(maxlen=20)
green_buoy_history = deque(maxlen=20)

def update_history(buoy_history, positions):
    for pos in positions:
        buoy_history.append(pos)

def draw_predictions(frame, detections):
    """Draws bounding boxes and labels on the frame based on model predictions."""
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extracting the bounding box coordinates
        conf = box.conf.item()  # Extracting confidence as a float
        class_id = box.cls.item()  # Extracting class ID as a float
        label = model.names[int(class_id)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        label_text = f"{label} {conf:.2f}"
        cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

def predict_and_draw_path(frame, red_history, green_history):
    """Draws a shaded path between the positions of red and green buoys."""
    print("Predict and draw path function called")
    if len(red_history) > 1 and len(green_history) > 1:
        red_points = np.array([[int(x), int(y)] for x, y in red_history], np.int32)
        green_points = np.array([[int(x), int(y)] for x, y in green_history], np.int32)
        
        # Sort points by y-coordinate to maintain top-to-bottom order
        red_points = red_points[np.argsort(red_points[:, 1])]
        green_points = green_points[np.argsort(green_points[:, 1])]

        print("Red points:", red_points)
        print("Green points:", green_points)
        
        if len(red_points) > 1 and len(green_points) > 1:
            path_points = np.vstack((red_points, green_points[::-1]))  # Combine and ensure proper ordering
            print("Path points:", path_points)
            
            # Create an overlay and draw the path
            overlay = frame.copy()
            cv2.fillPoly(overlay, [path_points], (255, 255, 0), lineType=cv2.LINE_AA)
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            print("Not enough points to form a path")
    else:
        print(f"Insufficient history: red_history ({len(red_history)}), green_history ({len(green_history)})")

def calculate_steering_command(frame, red_history, green_history):
    """Calculate steering command based on the midpoint between the latest red and green buoy positions."""
    if red_history and green_history:
        red_point = red_history[-1]
        green_point = green_history[-1]
        
        midpoint_x = (red_point[0] + green_point[0]) / 2
        frame_center_x = frame.shape[1] / 2

        if midpoint_x < frame_center_x - 10:  # Threshold for 'left' command
            command = "Left"
        elif midpoint_x > frame_center_x + 10:  # Threshold for 'right' command
            command = "Right"
        else:
            command = "Straight"

        return command
    return "No Command"

def process_frame(frame):
    """Process a single frame for object detection and annotation."""
    results = model(frame, imgsz=1280)[0]
    detections = results.boxes  # Extract detections
    return detections

def main():
    cap = cv2.VideoCapture('/Users/dinadehaini/Downloads/IMG_4502.mov')  # Ensure correct video path 2770
    #cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("Error: Video file not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        detections = process_frame(frame)
        draw_predictions(frame, detections)

        red_positions = []
        green_positions = []
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            class_id = box.cls.item()
            if class_id == 4:  # Update to match the class ID for red buoys
                red_positions.append((x_center, y_center))
            elif class_id == 2:  # Update to match the class ID for green buoys
                green_positions.append((x_center, y_center))

        print(f"Red positions: {red_positions}")
        print(f"Green positions: {green_positions}")

        update_history(red_buoy_history, red_positions)
        update_history(green_buoy_history, green_positions)
        
        predict_and_draw_path(frame, red_buoy_history, green_buoy_history)

        # Calculate and display steering command
        steering_command = calculate_steering_command(frame, red_buoy_history, green_buoy_history)
        cv2.putText(frame, f"Steering: {steering_command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('RoboBoat Buoy Navigation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



# import cv2
# import numpy as np
# import torch
# from collections import deque


# # Load the model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/dinadehaini/best (1).pt', force_reload=True, trust_repo=True)

# # History queues to track past positions of buoys
# red_buoy_history = deque(maxlen=20)
# green_buoy_history = deque(maxlen=20)

# def update_history(buoy_history, new_position):
#     """Updates the tracking history of a detected object."""
#     buoy_history.append(new_position)

# def predict_and_draw_path(frame, red_history, green_history):
#     """Draws a shaded path between the latest positions of red and green buoys."""
#     if len(red_history) > 1 and len(green_history) > 1:
#         # Create points for the polygon from red and green buoy histories
#         red_points = np.array([[int(x), int(y)] for x, y in red_history], np.int32)
#         green_points = np.array([[int(x), int(y)] for x, y in reversed(green_history)], np.int32)  # Reverse for correct polygon formation
#         path_points = np.vstack((red_points, green_points))  # Combine points

#         # Draw semi-transparent filled polygon
#         overlay = frame.copy()
#         cv2.fillPoly(overlay, [path_points], (255, 255, 0), lineType=cv2.LINE_AA)
#         alpha = 0.3  # Transparency factor
#         frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# def draw_predictions(frame, results):
#     """Draws bounding boxes and labels on the frame based on model predictions."""
#     results_data = results.pandas().xyxy[0].to_dict(orient="records")
#     for result in results_data:
#         x1, y1, x2, y2 = int(result['xmin']), int(result['ymin']), int(result['xmax']), int(result['ymax'])
#         class_label = result['name']
#         confidence = result['confidence']
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label_text = f"{class_label} ({confidence:.2f})"
#         cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


# def predict_and_draw_path(frame, red_history, green_history):
#     """Draws a shaded path between the latest positions of red and green buoys."""
#     if len(red_history) >= 1 and len(green_history) >= 1:
#         # Prepare points for the polygon
#         red_points = np.array([list(map(int, pos)) for pos in red_history], np.int32)
#         green_points = np.array([list(map(int, pos)) for pos in reversed(green_history)], np.int32)
        
#         if len(red_points) > 1 and len(green_points) > 1:
#             path_points = np.vstack((red_points, green_points))  # Combine points
#             path_points = path_points.reshape((-1, 1, 2))

#             # Draw semi-transparent filled polygon
#             overlay = frame.copy()
#             cv2.fillPoly(overlay, [path_points], (255, 255, 0), lineType=cv2.LINE_AA)
#             alpha = 0.3  # Transparency factor
#             frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
#         else:
#             print("Not enough points to form a path.")
#     else:
#         print("Insufficient history for both buoy types.")


# def predict_and_draw_path(frame, red_history, green_history):
#     """Draws a shaded path between the positions of red and green buoys."""
#     if len(red_history) >= 1 and len(green_history) >= 1:
#         # Prepare points for the polygon
#         red_points = np.array([list(map(int, pos)) for pos in red_history], np.int32)
#         green_points = np.array([list(map(int, pos)) for pos in reversed(green_history)], np.int32)
        
#         # Check if there are enough points to form a complex path
#         if len(red_points) > 1 and len(green_points) > 1:
#             path_points = np.vstack((red_points, green_points))  # Combine points
#             path_points = path_points.reshape((-1, 1, 2))
#         else:
#             # Not enough points for a complex path, draw a simple straight path
#             path_points = np.array([red_points[0], green_points[0]], np.int32)
#             path_points = path_points.reshape((-1, 1, 2))

#         # Draw semi-transparent filled polygon
#         overlay = frame.copy()
#         cv2.fillPoly(overlay, [path_points], (255, 255, 0), lineType=cv2.LINE_AA)
#         alpha = 0.4  # Transparency factor
#         frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
#     else:
#         print("Insufficient history for both buoy types - need at least one of each.")

# def main():
#     cap = cv2.VideoCapture('/Users/dinadehaini/Downloads/IMG_2770.MOV')
#     if not cap.isOpened():
#         print("Error: Video file not accessible")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("No more frames to read")
#             break

#         results = model(frame)
#         draw_predictions(frame, results)

#         red_positions = []
#         green_positions = []
#         for result in results.xyxy[0].numpy():
#             x_center = int((result[0] + result[2]) / 2)
#             y_center = int((result[1] + result[3]) / 2)
#             class_label = int(result[5])

#             if class_label == 0:  # Red buoys
#                 red_positions.append((x_center, y_center))
#             elif class_label == 1:  # Green buoys
#                 green_positions.append((x_center, y_center))

#         predict_and_draw_path(frame, red_positions, green_positions)

#         cv2.imshow('RoboBoat Buoy Navigation', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()


# # cap = cv2.VideoCapture('/Users/dinadehaini/Downloads/IMG_2770.MOV')
# # def main():
# #     cap = cv2.VideoCapture(1)
# #     if not cap.isOpened():
# #         print("Error: Webcam not accessible")
# #         return

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             print("Failed to grab frame")
# #             break
        
# #         results = model(frame)
        
# #         draw_predictions(frame, results)

# #         if len(results.xyxy[0]) > 0:
# #             for result in results.xyxy[0].numpy():
# #                 x_center = (result[0] + result[2]) / 2
# #                 y_center = (result[1] + result[3]) / 2
# #                 class_label = int(result[5])

# #                 if class_label == 0:  # Assuming class 0 is red buoys
# #                     update_history(red_buoy_history, (x_center, y_center))
# #                 elif class_label == 1:  # Assuming class 1 is green buoys
# #                     update_history(green_buoy_history, (x_center, y_center))

# #         draw_shaded_path(frame, red_buoy_history, (0, 0, 255))  # Semi-transparent red for red buoys
# #         draw_shaded_path(frame, green_buoy_history, (0, 255, 0))  # Semi-transparent green for green buoys
# #         red_path = predict_path(red_buoy_history, frame)
# #         green_path = predict_path(green_buoy_history, frame)
# #         command = adaptive_steering(red_path, green_path, frame.shape[1])
# #         display_steering_command(frame, command)

# #         cv2.imshow('RoboBoat Buoy Navigation', frame)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # if __name__ == '__main__':
# #     main()
