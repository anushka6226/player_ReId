from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO player detector
detector = YOLO('models/player_detector.pt')

# Initialize DeepSort with more forgiving settings
tracker = DeepSort(max_age=120, n_init=2, max_iou_distance=0.9)

# Open video file
cap = cv2.VideoCapture('15sec_input_720p.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare output writer
out = cv2.VideoWriter('outputs/output_with_ids.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

# Map DeepSort IDs to serial IDs
id_map = {}
next_serial_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame)

    detections = []
    for box, conf, cls in zip(results[0].boxes.xywh.cpu().numpy(),
                              results[0].boxes.conf.cpu().numpy(),
                              results[0].boxes.cls.cpu().numpy()):
        if int(cls) == 0 and conf > 0.3:
            x, y, w, h = box
            detections.append(([x, y, w, h], conf, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # Assign serial ID if new
        if track_id not in id_map:
            id_map[track_id] = next_serial_id
            next_serial_id += 1

        serial_id = id_map[track_id]

        l, t, r, b = track.to_ltrb()

        # Draw box
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)

        # Draw ID close to player
        cv2.putText(frame, f'ID {serial_id}', (int(l), int(b) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Player Re-ID', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
