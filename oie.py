import cv2
from ultralytics import YOLO

model_path = '/mnt/data/weights/runs/train/exp/weights/best.pt'

model = YOLO(model_path)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
    cv2.imshow('Camera com deteccao', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
