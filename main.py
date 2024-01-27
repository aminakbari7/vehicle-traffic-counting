
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import datetime

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("cars.mp4")

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

def main():
 writer = create_video_writer(cap, "00.mp4")
 while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    start = datetime.datetime.now()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        c=0
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)
            
        if boxes is not None:
         for box, track_id in zip(boxes, track_ids):
             x, y, w, h = box
             cv2.putText(frame,str(track_id), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.6, 255)
             c+=1
        cv2.putText(frame,"cars in image : "+str(c),(0,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255))

        # Display the annotated frame
        end = datetime.datetime.now()
        total = (end - start).total_seconds()
        fps = f"FPS: {1 / total:.2f}"
        writer.write(frame)
        cv2.imshow("YOLOv8 Tracking",frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
 cap.release()
 cv2.destroyAllWindows()

if __name__=="__main__":
    main()





