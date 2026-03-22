import time
import sys
import signal
import sqlite3
import json
from datetime import datetime
import os
import builtins
import threading
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import serialize_message


class Logger(Node):
    def __init__(self, logging_path, hz=30.0, batch_size=100, flush_interval=5.0):
        super().__init__("logging_node")

        self.prev_time = {}
        self.logging_path = logging_path
        self.hz = hz
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.conn = None
        self.cursor = None
        self.buffer = []
        self.last_flush_time = time.monotonic()

        self._original_print = builtins.print
        self.subscribers = {}
        self.db_lock = threading.Lock()

        self.save_dir = None
        self.file_index = {}
        self.raw_topic_names = set()

    def _log_print(self, *args, **kwargs):
        msg = " ".join(str(arg) for arg in args)
        timestamp = datetime.now().isoformat()

        self._original_print(f"[{timestamp}]", *args, **kwargs)

        if self.conn is not None:
            self.write_log(
                name="print",
                type_="stdout",
                data={
                    "timestamp": timestamp,
                    "message": msg
                },
                hz=None
            )

    def logging_start(self):
        builtins.print = self._log_print
        signal.signal(signal.SIGINT, self.signal_handler)

        folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join(self.logging_path, folder_name)
        os.makedirs(self.save_dir, exist_ok=True)

        db_path = os.path.join(self.save_dir, "log.db")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.db_lock = threading.Lock()

        self.cursor.execute("PRAGMA journal_mode=WAL")
        self.cursor.execute("PRAGMA synchronous=NORMAL")

        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def _topic_to_filename_prefix(self, topic_name):
        return topic_name.strip("/").replace("/", "_")

    def _save_raw_message(self, topic_name, msg):
        prefix = self._topic_to_filename_prefix(topic_name)
        topic_dir = os.path.join(self.save_dir, prefix)
        os.makedirs(topic_dir, exist_ok=True)
    
        with self.db_lock:
            idx = self.file_index.get(topic_name, 0)
            self.file_index[topic_name] = idx + 1
    
        file_path = os.path.join(topic_dir, f"{prefix}_{idx}")
    
        with open(file_path, "wb") as f:
            f.write(serialize_message(msg))
    
        rel_path = os.path.relpath(file_path, self.logging_path)
        return rel_path

    def write_log(self, name, type_, data, hz=None):
        if self.conn is None:
            return

        now = time.monotonic()

        if hz is not None:
            prev = self.prev_time.get(name, 0.0)
            if now - prev < 1.0 / hz:
                return

        row = (
            name,
            type_,
            datetime.now().isoformat(),
            json.dumps(data, ensure_ascii=False)
        )

        with self.db_lock:
            self.buffer.append(row)
            self.prev_time[name] = now

            if len(self.buffer) >= self.batch_size or now - self.last_flush_time >= self.flush_interval:
                self.cursor.executemany("""
                    INSERT INTO logs (name, type, timestamp, data)
                    VALUES (?, ?, ?, ?)
                """, self.buffer)
                self.conn.commit()
                self.buffer.clear()
                self.last_flush_time = time.monotonic()

    def flush(self):
        if not self.buffer or self.conn is None:
            return

        with self.db_lock:
            self.cursor.executemany("""
                INSERT INTO logs (name, type, timestamp, data)
                VALUES (?, ?, ?, ?)
            """, self.buffer)
            self.conn.commit()
            self.buffer.clear()
            self.last_flush_time = time.monotonic()

    def logging_terminate(self):
        if self.conn:
            self.flush()
            self.conn.close()
            self.conn = None
            self.cursor = None

        builtins.print = self._original_print
        print("#### log is saved")

    def signal_handler(self, sig, frame):
        self._original_print("\nCtrl+C was pressed. Exiting gracefully...")
        self.logging_terminate()
        rclpy.shutdown()
        sys.exit(0)

    def set_ros2_topic(self, topic_name, topic_type_str, hz=None, qos_profile=10, raw=False):
        msg_type = get_message(topic_type_str)

        if raw:
            self.raw_topic_names.add(topic_name)

        def callback(msg):
            if topic_name in self.raw_topic_names:
                data = self._save_raw_message(topic_name, msg)
            else:
                data = self._msg_to_dict(msg)

            self.write_log(
                name=topic_name,
                type_=topic_type_str,
                data=data,
                hz=hz
            )

        sub = self.create_subscription(
            msg_type,
            topic_name,
            callback,
            qos_profile
        )
        self.subscribers[topic_name] = sub
        print(f"subscriber created: {topic_name} [{topic_type_str}] raw={raw}")

    def _msg_to_dict(self, msg):
        if hasattr(msg, "__slots__"):
            result = {}
            for field in msg.__slots__:
                key = field.lstrip("_")
                value = getattr(msg, field)
                result[key] = self._convert_value(value)
            return result
        return str(msg)

    def _convert_value(self, value):
        if isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]

        if hasattr(value, "__slots__"):
            result = {}
            for field in value.__slots__:
                key = field.lstrip("_")
                result[key] = self._convert_value(getattr(value, field))
            return result

        return str(value)
