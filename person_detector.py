import cv2
import numpy as np

class PersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def count_people(self, frame: np.ndarray, people_coords):
        """Detect people and draw bounding boxes."""
        count = len(people_coords)
        for i, (x, y) in enumerate(people_coords):
            cv2.rectangle(frame, (x, y), (x + 20, y + 40), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return count

    def visualize(self, frame: np.ndarray, count: int, state):
        """Display drone simulation and camera view"""
        top_view = frame.copy()
        cv2.putText(top_view, f"Drone Position: {state['position']}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(top_view, f"People Count: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined_view = np.vstack([top_view, frame])
        cv2.imshow("Simulation", combined_view)
        cv2.waitKey(1)
