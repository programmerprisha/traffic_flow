import cv2
import numpy as np

# Video settings
width, height = 1280, 720
fps = 30
duration = 30  
frames = fps * duration

out = cv2.VideoWriter("traffic_sim.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
num_cars = 20
cars = []

for _ in range(num_cars):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    speed = np.random.randint(2, 8)
    cars.append([x, y, speed])

for _ in range(frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw road
    frame[:] = (50, 50, 50)

    for car in cars:
        car[0] += car[2]

        # Wrap around
        if car[0] > width:
            car[0] = 0

        cv2.rectangle(frame,
                      (car[0], car[1]),
                      (car[0] + 40, car[1] + 20),
                      (0, 255, 255),
                      -1)

    out.write(frame)

out.release()
print("Saved as traffic_sim.mp4")