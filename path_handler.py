import heapq
import logging
import math
import random
import time

from vision import VisionSystemSync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathPlanner:
    def __init__(self):
        self.known_obstacles = [
            (10, 0),
            (-10, -10),
            (0, 10),
            (15, 5),
            (-12, 12),
            (5, -15),
            (-8, -5),
            (20, 20),
            (-18, -3),
            (13, -7),
            (-7, 8),
            (18, -10),
            (-5, 17),
            (12, 13),
            (-16, -14),
            (3, -12),
            (-14, 0),
            (7, 16),
            (-20, 10),
            (0, -20),
            (4, 4),
        ]
        self.obstacle_radius = 4
        self.safety_margin = 3

    def is_position_safe(self, x, z):
        for obs_x, obs_z in self.known_obstacles:
            distance = math.sqrt((x - obs_x) ** 2 + (z - obs_z) ** 2)
            if distance < (self.obstacle_radius + self.safety_margin):
                return False
        return True

    def get_neighbors(self, pos, step_size=2):
        x, z = pos
        neighbors = []

        directions = [
            (0, step_size),  # North
            (step_size, step_size),  # Northeast
            (step_size, 0),  # East
            (step_size, -step_size),  # Southeast
            (0, -step_size),  # South
            (-step_size, -step_size),  # Southwest
            (-step_size, 0),  # West
            (-step_size, step_size),  # Northwest
        ]

        for dx, dz in directions:
            new_x, new_z = x + dx, z + dz
            if self.is_position_safe(new_x, new_z):
                neighbors.append((new_x, new_z))

        return neighbors

    def heuristic(self, pos1, pos2):
        """Calculate heuristic distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def a_star_search(self, start, goal):
        logger.info(f"Planning path from {start} to {goal}")

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        visited = set()

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current in visited:
                continue
            visited.add(current)

            if self.heuristic(current, goal) < 3:
                path = [goal]
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                logger.info(f"Path found with {len(path)} waypoints")
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                tentative_g = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        logger.warning("No path found with A*")
        return None

    def simple_greedy_path(self, start, goal, max_attempts=50):
        logger.info("Using simple greedy pathfinding")

        current = start
        path = [current]

        for _ in range(max_attempts):
            if self.heuristic(current, goal) < 3:
                path.append(goal)
                return path

            neighbors = self.get_neighbors(current)
            if not neighbors:
                break

            best_neighbor = min(neighbors, key=lambda n: self.heuristic(n, goal))

            if len(path) > 1 and best_neighbor == path[-2]:
                neighbors.remove(best_neighbor)
                if neighbors:
                    best_neighbor = min(
                        neighbors, key=lambda n: self.heuristic(n, goal)
                    )
                else:
                    break

            path.append(best_neighbor)
            current = best_neighbor

        return path if len(path) > 1 else None


class AutonomousNavigator:
    def __init__(self, goal_x=25, goal_z=25):
        self.vision_system = VisionSystemSync()
        self.path_planner = PathPlanner()
        self.goal = (goal_x, goal_z)
        self.current_path = []
        self.path_index = 0
        self.stuck_count = 0
        self.max_stuck_attempts = 3

    def start_navigation(self):
        logger.info("Starting autonomous navigation...")

        if not self.vision_system.start():
            logger.error("Failed to connect to robot")
            return False

        logger.info("Connected to robot successfully")

        try:
            return self.navigation_loop()
        except KeyboardInterrupt:
            logger.info("Navigation interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return False
        finally:
            self.vision_system.stop()

    def navigation_loop(self):
        """Main navigation loop"""
        while True:
            current_pos = self.get_current_position()
            logger.info(f"Current position: {current_pos}")

            if self.is_goal_reached(current_pos):
                logger.info(" Goal reached successfully!")
                return True

            analysis = self.vision_system.capture_and_analyze()
            if not analysis:
                logger.warning("Failed to analyze environment, using basic navigation")
                self.basic_movement_toward_goal(current_pos)
                continue

            logger.info(f"Environment analysis: {analysis['movement_advice']}")

            path = self.path_planner.a_star_search(current_pos, self.goal)
            if not path:
                path = self.path_planner.simple_greedy_path(current_pos, self.goal)

            if path and len(path) > 1:
                success = self.execute_path_step(path, analysis)
                if not success:
                    self.handle_navigation_failure()
            else:
                self.vision_based_navigation(analysis)

            time.sleep(1)

    def execute_path_step(self, path, analysis):
        if len(path) < 2:
            return False

        current = path[0]
        next_waypoint = path[1]

        dx = next_waypoint[0] - current[0]
        dz = next_waypoint[1] - current[1]
        distance = math.sqrt(dx**2 + dz**2)

        if distance < 1:
            return True

        angle = math.atan2(dx, dz) * 180 / math.pi

        move_distance = min(distance, 5)

        logger.info(f"Moving: turn {angle:.1f}°, distance {move_distance:.1f}")

        success = self.vision_system.move_robot_relative(angle, move_distance)

        if success:
            time.sleep(3)

            if self.vision_system.is_collision_detected():
                logger.warning("Collision detected during movement!")
                self.vision_system.reset_collision_flag()
                return False

            self.stuck_count = 0
            return True

        return False

    def vision_based_navigation(self, analysis):
        advice = analysis["movement_advice"]

        if advice == "forward":
            self.vision_system.move_robot_relative(0, 4)
        elif advice == "left":
            self.vision_system.move_robot_relative(-45, 3)
        elif advice == "right":
            self.vision_system.move_robot_relative(45, 3)
        elif advice == "backward":
            self.vision_system.move_robot_relative(180, 2)

        time.sleep(3)

        if self.vision_system.is_collision_detected():
            logger.warning("Collision during vision-based movement")
            self.vision_system.reset_collision_flag()
            return False

        return True

    def basic_movement_toward_goal(self, current_pos):
        dx = self.goal[0] - current_pos[0]
        dz = self.goal[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dz**2)

        if distance < 2:
            return True

        angle = math.atan2(dx, dz) * 180 / math.pi
        move_distance = min(distance, 3)

        logger.info(f"Basic movement toward goal: {angle:.1f}°, {move_distance:.1f}")

        self.vision_system.move_robot_relative(angle, move_distance)
        time.sleep(3)

        return not self.vision_system.is_collision_detected()

    def handle_navigation_failure(self):
        self.stuck_count += 1

        if self.stuck_count >= self.max_stuck_attempts:
            logger.warning("Robot seems stuck, attempting random exploration")
            self.random_exploration()
            self.stuck_count = 0

    def random_exploration(self):
        angles = [-90, -45, 45, 90, 135, -135]
        angle = random.choice(angles)
        distance = random.uniform(2, 4)

        logger.info(f"Random exploration: {angle}°, {distance:.1f}")

        self.vision_system.move_robot_relative(angle, distance)
        time.sleep(3)

    def get_current_position(self):
        pos = self.vision_system.get_robot_position()
        return (pos["x"], pos["z"])

    def is_goal_reached(self, current_pos, threshold=3.0):
        """Check if robot has reached the goal"""
        distance = math.sqrt(
            (current_pos[0] - self.goal[0]) ** 2 + (current_pos[1] - self.goal[1]) ** 2
        )
        return distance < threshold


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Robot Navigation")
    parser.add_argument("--goal-x", type=float, default=25, help="Goal X coordinate")
    parser.add_argument("--goal-z", type=float, default=25, help="Goal Z coordinate")

    args = parser.parse_args()

    print("=" * 60)
    print("AUTONOMOUS ROBOT NAVIGATION")
    print("=" * 60)
    print(f"Goal: ({args.goal_x}, {args.goal_z})")
    print("Make sure server.py is running!")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    navigator = AutonomousNavigator(args.goal_x, args.goal_z)

    try:
        success = navigator.start_navigation()
        if success:
            print("Navigation completed successfully!")
        else:
            print("Navigation failed")
    except KeyboardInterrupt:
        print("\nNavigation stopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
