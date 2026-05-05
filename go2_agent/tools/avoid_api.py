import json
import os
import subprocess
import sys
import textwrap
import time as time_module

from langchain.agents import tool


REQUEST_TOPIC = "/api/obstacles_avoid/request"
ROS_PYTHON = os.getenv("ROS_PYTHON", "/usr/bin/python3.8")
ROS_SETUP_BASH = os.getenv(
    "ROS_SETUP_BASH",
    "/home/unitree/unitree_msg_ws/install/setup.bash",
)
WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avoid_api_worker.py")
MATCH_TIMEOUT_SECONDS = 3.0
PUBLISH_RATE_HZ = 10.0
ONCE_PUBLISH_SECONDS = 0.3

_worker_process = None


def request_message(api_id, parameter=None, noreply=True):
    if parameter is None:
        parameter = {}
    return {
        "api_id": int(api_id),
        "parameter": parameter,
        "noreply": bool(noreply),
    }


def avoid_move_message(x=0.0, y=0.0, yaw=0.0):
    return request_message(
        1003,
        {"x": float(x), "y": float(y), "yaw": float(yaw), "mode": 0},
        noreply=True,
    )


def avoid_switch_message(enable):
    return request_message(1001, {"enable": bool(enable)}, noreply=False)


def avoid_remote_api_message(enable):
    return request_message(
        1004,
        {"is_remote_commands_from_api": bool(enable)},
        noreply=False,
    )


def _start_worker():
    global _worker_process

    if _worker_process is not None and _worker_process.poll() is None:
        return _worker_process

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    _worker_process = subprocess.Popen(
        [
            "/bin/bash",
            "-lc",
            f"source {ROS_SETUP_BASH} && exec {ROS_PYTHON} {WORKER_PATH}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    return _worker_process


def _worker_request(payload):
    process = _start_worker()

    if process.stdin is None or process.stdout is None:
        return False, "Avoid worker pipes are unavailable."

    try:
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()
    except Exception as exc:
        return False, str(exc)

    line = process.stdout.readline()
    if not line:
        stderr_output = ""
        if process.stderr is not None:
            stderr_output = process.stderr.read().strip()
        return False, stderr_output or "Avoid worker returned no response."

    try:
        response = json.loads(line)
    except json.JSONDecodeError:
        return False, line.strip()

    if response.get("ok"):
        return True, response
    return False, response.get("error", "Unknown avoid worker error.")


def _publish_message(message, duration_seconds):
    success, response = _worker_request(
        {
            "cmd": "publish",
            "message": message,
            "duration_seconds": float(duration_seconds),
            "rate_hz": PUBLISH_RATE_HZ,
            "match_timeout_seconds": MATCH_TIMEOUT_SECONDS,
            "topic_name": REQUEST_TOPIC,
        }
    )
    if success:
        return ""
    return response


def ros2_publish_once(message):
    return _publish_message(message, ONCE_PUBLISH_SECONDS)


def ros2_publish_for(message, seconds):
    return _publish_message(message, float(seconds))


def _ready_avoid():
    error = ros2_publish_once(avoid_switch_message(True))
    if error:
        return error
    error = ros2_publish_once(avoid_remote_api_message(True))
    if error:
        return error
    return "Avoid API is ready. switch=True remote_command=True"


@tool
def ready_avoid() -> str:
    """
    Prepare obstacle avoidance API control.
    Use this only when the user explicitly asks to prepare or recover avoid mode.
    """
    return _ready_avoid()


@tool
def avoid_api_move(x: float, y: float, z: float, time: float) -> str:
    """
    Move Go2 with obstacle avoidance API for a short duration.
    x: forward velocity
    y: lateral velocity
    z: yaw velocity
    time: movement duration in seconds
    """
    seconds = float(time)
    if seconds <= 0.0:
        return "time must be positive seconds."

    error = ros2_publish_for(avoid_move_message(x, y, z), seconds)
    if error:
        return error
    error = ros2_publish_once(avoid_move_message(0.0, 0.0, 0.0))
    if error:
        return error
    return "Published avoid move command for %s seconds." % seconds


@tool
def avoid_api_stop() -> str:
    """
    Stop Go2 obstacle avoidance motion.
    """
    error = ros2_publish_once(avoid_move_message(0.0, 0.0, 0.0))
    return error or "Published avoid stop command."
