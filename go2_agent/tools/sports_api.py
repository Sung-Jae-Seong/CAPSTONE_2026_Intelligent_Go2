import json
import subprocess
import time as time_module

from langchain.agents import tool


MOVE_TOPIC = "/api/sport/request"
MOVE_TYPE = "unitree_api/msg/Request"
MOVE_RATE = "10"


def _move_message(x, y, z):
    parameter = json.dumps({"x": float(x), "y": float(y), "z": float(z)})
    return (
        "{header: {identity: {id: 0, api_id: 1008}, lease: {id: 0}, "
        "policy: {priority: 0, noreply: false}}, "
        f"parameter: '{parameter}', binary: []"
        "}"
    )


def _stop_message():
    return (
        "{header: {identity: {id: 0, api_id: 1003}, lease: {id: 0}, "
        "policy: {priority: 0, noreply: false}}, parameter: '', binary: []}"
    )


def _command_error(returncode, stdout="", stderr=""):
    if stderr and stderr.strip():
        return stderr.strip()
    if stdout and stdout.strip():
        return stdout.strip()
    return f"Command failed with code {returncode}."


def _sports_api_stop():
    result = subprocess.run(
        ["ros2", "topic", "pub", "-1", MOVE_TOPIC, MOVE_TYPE, _stop_message()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return _command_error(result.returncode, result.stdout, result.stderr)
    return ""


@tool
def sports_api_move(x: float, y: float, z: float, time: float) -> str:
    """
    Move Go2 with sport API for a short duration.
    x: forward velocity
    y: lateral velocity
    z: yaw velocity
    time: publish duration in seconds
    """
    process = subprocess.Popen(
        ["ros2", "topic", "pub", "-r", MOVE_RATE, MOVE_TOPIC, MOVE_TYPE, _move_message(x, y, z)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time_module.sleep(time)
    if process.poll() not in (None, 0):
        stdout, stderr = process.communicate()
        return _command_error(process.returncode, stdout, stderr)
    process.terminate()
    process.wait(timeout=1)
    if process.returncode not in (0, -15):
        stdout, stderr = process.communicate()
        return _command_error(process.returncode, stdout, stderr)
    error = _sports_api_stop()
    if error:
        return error
    return f"Published move command for {time} seconds."


@tool
def sports_api_stop() -> str:
    """
    Stop Go2.
    """
    error = _sports_api_stop()
    if error:
        return error
    return "Published stop command."
