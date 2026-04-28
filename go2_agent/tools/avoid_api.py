import json
import shlex
import subprocess
import time as time_module

from langchain.agents import tool

from .ros2 import execute_ros_command


def request_message(api_id, parameter=None, noreply=True):
    if parameter is None:
        parameter = {}
    parameter = json.dumps(parameter, separators=(",", ":"))
    return (
        "{header: {identity: {id: %d, api_id: %d}, lease: {id: 0}, "
        "policy: {priority: 0, noreply: %s}}, "
        "parameter: '%s', binary: []}"
    ) % (
        time_module.monotonic_ns(),
        api_id,
        "true" if noreply else "false",
        parameter,
    )


def avoid_move_message(x=0.0, y=0.0, yaw=0.0):
    return request_message(
        1003,
        {"x": float(x), "y": float(y), "yaw": float(yaw), "mode": 0},
        noreply=True,
    )


def avoid_switch_message(enable):
    return request_message(1001, {"enable": bool(enable)}, noreply=False)


def avoid_switch_get_message():
    return request_message(1002, {}, noreply=False)


def avoid_remote_api_message(enable):
    return request_message(
        1004,
        {"is_remote_commands_from_api": bool(enable)},
        noreply=False,
    )


def topic_pub_command(extra_args, message):
    return "ros2 topic pub %s /api/obstacles_avoid/request unitree_api/msg/Request %s" % (
        extra_args,
        shlex.quote(message),
    )


def ros2_publish_once(message):
    success, output = execute_ros_command(topic_pub_command("--once", message))
    return "" if success else output


def ros2_publish_for(message, seconds):
    process = subprocess.Popen(
        shlex.split(topic_pub_command("-r 10", message)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time_module.sleep(seconds)
    finally:
        if process.poll() is None:
            process.terminate()
    stdout, stderr = process.communicate(timeout=1)
    if process.returncode not in (0, -15):
        return stderr.strip() or stdout.strip() or "Command failed."
    return ""


def read_avoid_response(message):
    echo = subprocess.Popen(
        [
            "ros2",
            "topic",
            "echo",
            "--qos-reliability",
            "best_effort",
            "/api/obstacles_avoid/response",
            "unitree_api/msg/Response",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time_module.sleep(0.1)
    ros2_publish_once(message)
    try:
        stdout, _stderr = echo.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        echo.terminate()
        stdout, _stderr = echo.communicate(timeout=1)
    return stdout


def avoid_switch_enabled():
    output = read_avoid_response(avoid_switch_get_message())
    if '"enable":true' in output or '\\"enable\\":true' in output:
        return True
    if '"enable":false' in output or '\\"enable\\":false' in output:
        return False
    return None


def _ready_avoid():
    enabled = avoid_switch_enabled()
    if enabled is not True:
        error = ros2_publish_once(avoid_switch_message(True))
        if error:
            return error
    error = ros2_publish_once(avoid_remote_api_message(True))
    if error:
        return error
    return "Avoid API is ready. switch=%s remote_command=True" % enabled


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
