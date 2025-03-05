import cv2
import numpy as np

class CameraSimulator:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height

    def simulate_frame(self, position: np.ndarray, orientation: np.ndarray):
        """Simulate drone and camera environment"""
        frame = np.full((self.height, self.width, 3), (135, 206, 235), dtype=np.uint8)  # Sky blue background
        ground_y = int(self.height * 0.75)
        cv2.rectangle(frame, (0, ground_y), (self.width, self.height), (50, 150, 50), -1)  # Green ground

        # Drone motion visualization
        cx, cy = int(self.width / 2 + position[0] * 20), int(self.height / 3 + position[1] * 20)
        cv2.circle(frame, (cx, cy), 40, (0, 0, 255), -1)  # Red body
        cv2.putText(frame, "Hexacopter", (cx - 50, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Rotors
        rotor_positions = [(-50, -30), (50, -30), (-60, 10), (60, 10), (-40, 40), (40, 40)]
        for rx, ry in rotor_positions:
            cv2.circle(frame, (cx + rx, cy + ry), 12, (255, 255, 255), -1)

        # People (0-3)
        num_people = np.random.randint(0, 4)
        people_coords = []
        for i in range(num_people):
            x, y = np.random.randint(50, self.width - 50), np.random.randint(ground_y + 10, self.height - 50)
            cv2.rectangle(frame, (x, y), (x + 20, y + 40), (0, 0, 255), -1)
            cv2.putText(frame, f"Person {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            people_coords.append((x, y))

        return frame, people_coords
