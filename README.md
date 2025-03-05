# Hexacopter Drone Simulation

## Overview
This project simulates a hexacopter drone in motion, using **Pixhawk 2.4.8** for flight control (or a simulated alternative). The simulation includes a **dual-view visualization**:
1. **Drone motion view** - Shows the hexacopter moving in a simulated 3D-like environment.
2. **Camera output view** - Displays the drone's camera feed with people detection and labels.

## Features
- **Hexacopter motion simulation** (realistic movement based on Pixhawk commands or random actions)
- **People detection** (0-3 people randomly placed in the environment)
- **Dual visualization** (Drone motion + Camera feed)
- **Real-time labels** (Drone, Rotors, People, Ground)
- **Logs recorded per iteration** (`log.txt` with drone position & people count)

## Files and Their Purpose
### 1️⃣ `main.py`
- Entry point for the simulation.
- Initializes Pixhawk 2.4.8 or a simulated interface.
- Runs the drone's movement and captures camera frames.
- Saves logs to `log.txt`.

### 2️⃣ `camera_simulator.py`
- Generates a **3D-like environment** with a sky and ground.
- Simulates the hexacopter's **motion and position**.
- Places **0-3 people** randomly in the environment.

### 3️⃣ `person_detector.py`
- Uses **HOG + SVM-based detection** to recognize people.
- Draws **bounding boxes** around detected people.
- Displays **camera output** with the detected people count.

### 4️⃣ `pixhawk_interface.py`
- Connects to **Pixhawk 2.4.8** via MAVLink.
- Fetches real drone movement data.
- If Pixhawk is unavailable, uses a **simulated movement generator**.

### 5️⃣ `environment.py`
- Defines the **simulation physics**.
- Updates drone state (position, velocity, attitude) over time.
- Ensures smooth physics transitions for motion.

### 6️⃣ `log.txt`
- Stores logs for each simulation step.
- Saves **drone position, step count, and detected people**.
- Used for debugging and performance tracking.

## Installation
### 1️⃣ Install Dependencies
Run the following command:
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Simulation
```bash
python main.py
```

## Expected Output
✅ **A hexacopter flying in a simulated 3D environment**  
✅ **People appearing in the scene (0-3), detected by the camera**  
✅ **Dual visualization: Drone motion + Camera feed**  
✅ **Real-time logs saved to `log.txt`**  

## Notes
- If you have **Pixhawk 2.4.8**, make sure it's connected before running.
- If Pixhawk is unavailable, the system will switch to **simulation mode**.
- Press **'q'** to exit the visualization.

## Author
- **Aarohi Singh: aarohi.pune@gmail.com** 🚀
