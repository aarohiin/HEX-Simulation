import numpy as np
import pymavlink
import serial
from pymavlink import mavutil

class PixhawkInterface:
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.vehicle = None

    def connect(self) -> bool:
        """Connect to Pixhawk"""
        try:
            self.vehicle = mavutil.mavlink_connection(serial.Serial(self.port, self.baud))
            self.vehicle.wait_heartbeat()
            print(f"Connected: System {self.vehicle.target_system}, Component {self.vehicle.target_component}")
            return True
        except Exception as e:
            print(f"Error connecting to Pixhawk: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect safely"""
        if self.vehicle:
            self.vehicle.close()
            self.vehicle = None

    def get_action(self) -> np.ndarray:
        """Get action command from Pixhawk"""
        try:
            msg = self.vehicle.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=0.1)
            if msg is None:
                return np.random.uniform(-1, 1, 6)

            position = np.array([msg.x, msg.y, msg.z])
            velocity = np.array([msg.vx, msg.vy, msg.vz])

            return np.clip(np.concatenate([position / 10.0, velocity / 5.0]), -1.0, 1.0)
        except Exception as e:
            print(f"Error reading Pixhawk data: {str(e)}")
            return np.random.uniform(-1, 1, 6)
