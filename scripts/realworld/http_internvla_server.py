import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

######## ros2 viz
import cv2
import rclpy
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
######## ros2 viz

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
######## ros2 viz
ros_node = None
viz_pub = None
bridge = CvBridge()
######## ros2 viz
idx = 0
start_time = time.time()
output_dir = ''

######## logging server
# import go2_monitor
# SOURCE = "zenoh"
# PORT = 8080
# ENDPOINT = "udp/192.168.0.151:7447"
# option = go2_monitor.monitor_option()
# option.source = SOURCE
# option.port = PORT
# option.endpoint = ENDPOINT
# go2_monitor.start_monitor(option)
########


######## instruction update
instruction = "follow the person"

@app.route("/update_instruction", methods=["POST"])
def update_instruction():
    global instruction

    new_instruction = request.get_data(as_text=True)
    if new_instruction and len(new_instruction) > 0:
        instruction = new_instruction.strip()
        return f"Successfully updated to: {new_instruction}\n", 200
    else:
        return "Error: Empty instruction\n", 400
######## instruction update

@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ######## instruction update
    print('[Current Instruction]', instruction[:10], '...')
    ######## instruction update
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    dual_sys_output, yolo_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output, yolo_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )

    json_output = {}
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
    else:
        json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel

    ######## ros2 viz
    if viz_pub is not None:
        try:
            viz_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

            # pixel goal viz (LLM 384x384 좌표계 → 원본 해상도로 매핑)
            orig_h, orig_w = viz_img.shape[:2]
            if dual_sys_output.output_pixel is not None:
                model_py = float(dual_sys_output.output_pixel[0])  # y in 384 space
                model_px = float(dual_sys_output.output_pixel[1])  # x in 384 space

                px = int((model_px / 384.0) * orig_w)
                py = int((model_py / 384.0) * orig_h)

                px = max(0, min(orig_w - 1, px))
                py = max(0, min(orig_h - 1, py))
                
                cv2.circle(viz_img, (px, py), radius=10, color=(0, 0, 255), thickness=-1)

            # YOLO bbox viz
            if yolo_output is not None and yolo_output.get('bbox') is not None:
                x1, y1, x2, y2 = yolo_output['bbox']
                conf = yolo_output['confidence']
                target = yolo_output['target']
                color = (0, 255, 0)  # 녹색 (threshold 이상, 사용됨)
                label = f"{target} {conf:.2f}"
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(viz_img, label, (x1, max(y1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            viz_img = cv2.resize(viz_img, (orig_w // 2, orig_h // 2))

            msg = bridge.cv2_to_imgmsg(viz_img, encoding="bgr8")
            viz_pub.publish(msg)
        except Exception as e:
            print(f"Failed to publish visualization: {e}")
    ######## ros2 viz
    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':
    project_path = '/'.join(os.path.abspath(__file__).split('/')[1:-3])
    model_path = f'/{project_path}/checkpoints/InternVLA-N1-DualVLN'

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    # parser.add_argument("--plan_step_gap", type=int, default=4)
    ############### !!!! 현재는 매 스텝마다 S2를 수행해서 유사 비동기도 아니고 동기인 상태!!
    ############### notebooks 폴더의 코드에서는 plan_step_gap=4 였음.
    ############### https://zhuanlan.zhihu.com/p/1969046543286907790
    ############### 위 링크에서 사용한 값 plan_step_gap=8 였음.
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [
        [905.1950073242188, 0.0, 670.1882934570312, 0.0],
        [0.0, 904.0057373046875, 385.49102783203125, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    agent = InternVLAN1AsyncAgent(args)
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.float32),
        np.eye(4, dtype=np.float32),
        "hello",
        intrinsic=args.camera_intrinsic,
    )
    agent.reset()

    ######## ros2 viz
    rclpy.init(args=None)
    ros_node = rclpy.create_node('internvla_viz_node')
    viz_pub = ros_node.create_publisher(ImageMsg, '/internvla/viz', 10)
    ######## ros2 viz

    try:
        app.run(host='0.0.0.0', port=5801)
    except KeyboardInterrupt:
        ...
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
