# Autonomous Robot Navigation System

This system implements autonomous collision-free route planning for a 3D robot simulator using computer vision and path planning algorithms.

## System Overview

The system consists of two main modules:

1. **vision.py** - Module for processing captured images and providing navigation instructions to the robot
2. **path_handler.py** - Autonomous route-finding logic with pathfinding algorithms

## Prerequisites

Make sure you have the following Python packages installed:

```bash
pip install opencv-python numpy pillow websockets asyncio requests heapq math
```

## Setup Instructions

1. **Start the Simulator Server**
   ```bash
   python server.py
   ```
   This will start:
   - WebSocket server on `ws://localhost:8080`
   - Flask API server on `http://localhost:5000`
   - Web interface at `http://localhost:5000`

2. **Open the Simulator**
   - Navigate to `http://localhost:5000` in your browser
   - You should see the 3D robot simulator with the robot at the origin
   - Green boxes represent obstacles in the environment

3. **Run the Autonomous Navigation**
   ```bash
   python path_handler.py
   ```
   OR use the launcher script:
   ```bash
   python run_autonomous.py --goal-x 25 --goal-z 25
   ```

## How It Works

### Vision System (`vision.py`)
- Connects to the WebSocket server to receive real-time data
- Captures images from the robot's camera using `/capture` API endpoint
- Processes images to detect obstacles and free paths
- Provides movement recommendations based on visual analysis
- Monitors for collision messages from the simulator

### Path Planning (`path_handler.py`)
- Implements A* algorithm for optimal pathfinding
- Implements RRT (Rapidly-exploring Random Tree) as backup
- Maintains a grid-based map of known obstacles
- Plans collision-free routes from current position to goal
- Executes autonomous navigation with real-time replanning

### Key Features
- **Fully Autonomous**: No manual input required after launch
- **Collision Avoidance**: Uses both visual processing and collision feedback
- **Adaptive Planning**: Switches between A* and RRT algorithms as needed
- **Real-time Replanning**: Updates path when obstacles are detected
- **WebSocket Communication**: Real-time bidirectional communication with simulator

## Usage

### Basic Usage
```bash
# Start with default goal (25, 25)
python path_handler.py
```

### Custom Goal Position
```bash
python run_autonomous.py --goal-x 30 --goal-z -15
```

## API Endpoints Used

The system interacts with these Flask endpoints:

- `POST /capture` - Capture image from robot camera
- `POST /move` - Move robot to absolute position (x, z coordinates)
- `POST /move_rel` - Move robot relative to current position (turn angle, distance)
- `POST /stop` - Stop robot movement

## WebSocket Messages

The system handles these WebSocket message types:

- `capture_image_response` - Contains base64 encoded image data
- `collision` - Collision detection notification
- `confirmation` - Movement completion confirmation

## Algorithm Details

### A* Pathfinding
- Grid-based pathfinding with 8-directional movement
- Uses Euclidean distance heuristic
- Optimal for known static environments

### RRT (Rapidly-exploring Random Tree)
- Sampling-based algorithm for complex environments
- Better for dynamic or unknown obstacles
- Falls back when A* fails

### Vision Processing
- HSV color space analysis for obstacle detection
- Sector-based analysis for movement direction recommendations
- Real-time image processing from robot's perspective

## Troubleshooting

### Common Issues

1. **"No connected simulators" error**
   - Ensure `server.py` is running
   - Check that WebSocket connection is established

2. **Robot doesn't move**
   - Verify Flask server is running on port 5000
   - Check browser console for WebSocket errors

3. **Path planning fails**
   - Goal position might be inside an obstacle
   - Try different goal coordinates

4. **Image processing errors**
   - Ensure OpenCV and PIL are properly installed
   - Check image capture is working via `/capture` endpoint

## Expected Behavior

When running successfully, you should see:

1. Robot starts at origin (0, 0, 0)
2. System captures images and analyzes environment
3. Path is planned avoiding green obstacle boxes
4. Robot moves autonomously toward goal
5. Real-time replanning occurs if collisions detected
6. Robot reaches goal without manual intervention
