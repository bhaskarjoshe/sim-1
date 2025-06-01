import asyncio
import base64
import io
import json
import logging
import threading
import time

import cv2
import numpy as np
import requests
import websockets
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionSystem:
    def __init__(
        self, websocket_url="ws://localhost:8080", flask_url="http://localhost:5000"
    ):
        self.ws_url = websocket_url
        self.flask_url = flask_url
        self.websocket = None
        self.current_image = None
        self.robot_position = {"x": 0, "y": 0, "z": 0}
        self.collision_detected = False
        self.image_received = False
        self.last_analysis = None
        self.connected = False
        self.loop = None

    async def connect_websocket(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.connected = True
            logger.info("Connected to robot simulator")

            await self.websocket.send(
                json.dumps({"type": "connection", "message": "Vision system connected"})
            )

            await self.listen_for_messages()

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False

    async def listen_for_messages(self):
        try:
            async for message in self.websocket:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")
            self.connected = False

    async def handle_message(self, message):
        try:
            data = json.loads(message)

            if data.get("type") == "capture_image_response":
                logger.info("Received image from robot")
                self.process_captured_image(data)

            elif data.get("type") == "collision":
                logger.warning("Collision detected!")
                self.collision_detected = True

            elif data.get("type") == "confirmation":
                logger.info(f"Robot confirmation: {data.get('message')}")
                if "position" in data:
                    self.robot_position = data["position"]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def process_captured_image(self, data):
        try:
            image_data = data["image"]

            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            self.current_image = cv_image

            if "position" in data:
                self.robot_position = data["position"]

            self.last_analysis = self.analyze_environment(cv_image)
            self.image_received = True

            logger.info("Image processed successfully")

        except Exception as e:
            logger.error(f"Error processing captured image: {e}")

    def analyze_environment(self, image):
        try:
            height, width = image.shape[:2]

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            bottom_half = green_mask[height // 2 :, :]

            sectors = self.analyze_movement_sectors(bottom_half)

            movement_advice = self.determine_movement_direction(sectors)

            return {
                "sectors": sectors,
                "movement_advice": movement_advice,
                "obstacle_detected": np.sum(green_mask) > 1000,
            }

        except Exception as e:
            logger.error(f"Error analyzing environment: {e}")
            return None

    def analyze_movement_sectors(self, obstacle_mask):
        height, width = obstacle_mask.shape

        left_sector = obstacle_mask[:, : width // 3]
        center_sector = obstacle_mask[:, width // 3 : 2 * width // 3]
        right_sector = obstacle_mask[:, 2 * width // 3 :]

        sectors = {
            "left": np.sum(left_sector) / (left_sector.size * 255),
            "center": np.sum(center_sector) / (center_sector.size * 255),
            "right": np.sum(right_sector) / (right_sector.size * 255),
        }

        return sectors

    def determine_movement_direction(self, sectors):
        obstacle_threshold = 0.05

        directions = {
            "forward": sectors["center"] < obstacle_threshold,
            "left": sectors["left"] < obstacle_threshold,
            "right": sectors["right"] < obstacle_threshold,
        }

        if directions["forward"]:
            return "forward"
        elif directions["left"] and directions["right"]:
            return "left"
        elif directions["left"]:
            return "left"
        elif directions["right"]:
            return "right"
        else:
            return "backward"

    def capture_image(self):
        try:
            self.image_received = False
            response = requests.post(f"{self.flask_url}/capture", timeout=5)

            if response.status_code == 200:
                logger.info("Image capture request sent")
                return True
            else:
                logger.error(f"Failed to capture image: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.error("Timeout sending capture request")
            return False
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return False

    def wait_for_image(self, timeout=5):
        start_time = time.time()
        while not self.image_received and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        return self.image_received

    def get_movement_instruction(self):
        if not self.last_analysis:
            return None

        advice = self.last_analysis["movement_advice"]

        if advice == "forward":
            return {"type": "move_rel", "turn": 0, "distance": 5}
        elif advice == "left":
            return {"type": "move_rel", "turn": -45, "distance": 3}
        elif advice == "right":
            return {"type": "move_rel", "turn": 45, "distance": 3}
        elif advice == "backward":
            return {"type": "move_rel", "turn": 180, "distance": 2}

        return None

    def move_robot(self, x, z):
        try:
            data = {"x": x, "z": z}
            response = requests.post(f"{self.flask_url}/move", json=data, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error moving robot: {e}")
            return False

    def move_robot_relative(self, turn, distance):
        try:
            data = {"turn": turn, "distance": distance}
            response = requests.post(f"{self.flask_url}/move_rel", json=data, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error moving robot relatively: {e}")
            return False

    def stop_robot(self):
        try:
            response = requests.post(f"{self.flask_url}/stop", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error stopping robot: {e}")
            return False

    def get_robot_position(self):
        return self.robot_position

    async def close(self):
        if self.websocket:
            await self.websocket.close()
        self.connected = False


class VisionSystemSync:
    def __init__(
        self, websocket_url="ws://localhost:8080", flask_url="http://localhost:5000"
    ):
        self.vision_system = VisionSystem(websocket_url, flask_url)
        self.loop = None
        self.thread = None

    def start(self):

        def run_async():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.vision_system.connect_websocket())

        self.thread = threading.Thread(target=run_async, daemon=True)
        self.thread.start()

        timeout = 10
        start_time = time.time()
        while not self.vision_system.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        return self.vision_system.connected

    def capture_and_analyze(self, timeout=10):
        if not self.vision_system.connected:
            logger.error("Not connected to robot")
            return None

        if not self.vision_system.capture_image():
            logger.error("Failed to send capture request")
            return None

        if not self.vision_system.wait_for_image(timeout):
            logger.error("Timeout waiting for image")
            return None

        return self.vision_system.last_analysis

    def get_movement_instruction(self):
        return self.vision_system.get_movement_instruction()

    def move_robot_relative(self, turn, distance):
        return self.vision_system.move_robot_relative(turn, distance)

    def move_robot(self, x, z):
        return self.vision_system.move_robot(x, z)

    def stop_robot(self):
        return self.vision_system.stop_robot()

    def get_robot_position(self):
        return self.vision_system.get_robot_position()

    def is_collision_detected(self):
        return self.vision_system.collision_detected

    def reset_collision_flag(self):
        self.vision_system.collision_detected = False

    def stop(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.vision_system.close(), self.loop)


def test_vision_system():
    print("Testing Vision System...")

    vision = VisionSystemSync()

    if not vision.start():
        print("Failed to connect to robot simulator")
        return False

    print("Connected to robot simulator")

    print("Capturing and analyzing image...")
    analysis = vision.capture_and_analyze()

    if analysis:
        print("Analysis successful!")
        print(f"Sectors: {analysis['sectors']}")
        print(f"Movement advice: {analysis['movement_advice']}")
        print(f"Obstacle detected: {analysis['obstacle_detected']}")

        instruction = vision.get_movement_instruction()
        if instruction:
            print(f"Movement instruction: {instruction}")

        return True
    else:
        print("Failed to analyze environment")
        return False


if __name__ == "__main__":
    test_vision_system()
    print("Vision system test complete")
