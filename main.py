import os
import logging
import cv2
import numpy as np
from dotenv import load_dotenv
from environment import HexacopterEnv
from camera_simulator import CameraSimulator
from person_detector import PersonDetector
from pixhawk_interface import PixhawkInterface

# Configure logging to console and file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_file = "log.txt"
if os.path.exists(log_file):
    os.remove(log_file)  # Clear log on restart
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

class SimulationManager:
    def __init__(self, simulate_pixhawk: bool = True):
        load_dotenv()
        self.env = HexacopterEnv()
        self.camera = CameraSimulator()
        self.detector = PersonDetector()
        self.pixhawk = self._initialize_pixhawk(simulate_pixhawk)
        self.episode_counter = 0
        self.total_reward = 0.0

    def _initialize_pixhawk(self, simulate: bool):
        if simulate:
            logger.info("Using simulated Pixhawk interface")
            return self._create_mock_pixhawk()
        
        try:
            pixhawk = PixhawkInterface(port=os.getenv('PIXHAWK_PORT', '/dev/ttyUSB0'),
                                       baud=int(os.getenv('PIXHAWK_BAUD', '115200')))
            if pixhawk.connect():
                return pixhawk
            logger.warning("Failed to connect to Pixhawk 2.4.8. Using simulation.")
        except Exception as e:
            logger.error(f"Pixhawk initialization error: {e}")
        return self._create_mock_pixhawk()

    def _create_mock_pixhawk(self):
        class MockPixhawk:
            def connect(self): return True
            def disconnect(self): pass
            def get_action(self): return np.random.uniform(-1, 1, 6)
        return MockPixhawk()

    def get_action(self):
        try:
            return self.pixhawk.get_action() if self.pixhawk else self.env.action_space.sample()
        except Exception as e:
            logger.warning(f"Action retrieval error: {e}")
            return self.env.action_space.sample()

    def run_simulation(self, max_episodes=3, max_steps_per_episode=500):
        try:
            for episode in range(max_episodes):
                obs, _ = self.env.reset()
                self.episode_counter += 1
                episode_reward = 0.0

                for step in range(max_steps_per_episode):
                    action = self.get_action()
                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_reward += reward

                    state = self.env.get_state()
                    frame, people_coords = self.camera.simulate_frame(state['position'], state['attitude'])
                    people_count = self.detector.count_people(frame, people_coords)

                    # Log details to file
                    with open(log_file, "a") as f:
                        f.write(f"Episode {episode+1}, Step {step}, People: {people_count}, Position: {state['position']}\n")

                    self.detector.visualize(frame, people_count, state)

                    if step % 50 == 0:
                        logger.info(f"Episode {episode+1}, Step {step}: Reward = {reward:.4f}, People Count: {people_count}")

                    if done or truncated:
                        logger.info(f"Episode {episode+1} ended. Total Reward: {episode_reward:.4f}")
                        break

                self.total_reward += episode_reward

            # Keep visualization open for 30 seconds
            cv2.waitKey(30000)
            cv2.destroyAllWindows()
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")

def main():
    simulation = SimulationManager(simulate_pixhawk=False)  # Set to True if Pixhawk isn't connected
    simulation.run_simulation()

if __name__ == "__main__":
    main()
