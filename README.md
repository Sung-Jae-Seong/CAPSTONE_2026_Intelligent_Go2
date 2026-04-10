# go2_monitor
`go2_monitor`는 ROS 2 Humble 기반 C++ 모니터링 서버다. 실행 파일은 `go2_monitor_cpp` 패키지의 `monitor_server`이며, 주 사용 모드는 다음 두 가지다.
- `--source=zenoh`: 실시간 Zenoh 스트림 뷰어
- `--source=/path/to/log.db`: SQLite 로그 재생 뷰어

## 환경
기준 환경:
- Ubuntu + ROS 2 Humble
- CMake 3.16+
- C++17
- `cargo 1.94.0`
- `rustc 1.94.0`
- `zenoh-c 1.8.0`
- `zenoh-cpp 1.8.0`
- `zenoh-bridge-ros2dds 1.8.0`

필수 패키지:
```bash
sudo apt update
sudo apt install -y \
  curl git cmake build-essential ninja-build pkg-config clang \
  libasio-dev libssl-dev zlib1g-dev \
  libsqlite3-dev libjpeg-dev libpng-dev \
  sqlite3 python3-pip python3-dev
```

## 의존성 설치
### 1. Rust
```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
rustup update
cargo --version
rustc --version
```
### 2. zenoh-c
```bash
mkdir -p ~/go2_monitor/zenoh
cd ~/go2_monitor/zenoh
git clone https://github.com/eclipse-zenoh/zenoh-c.git
cd zenoh-c
mkdir -p build
cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build .
cmake --install .
```
### 3. zenoh-cpp
```bash
cd ~/go2_monitor/zenoh
git clone https://github.com/eclipse-zenoh/zenoh-cpp.git
cd zenoh-cpp
mkdir -p build
cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build .
cmake --install .
```
### 4. Python 바인딩 빌드 의존성
```bash
python3 -m pip install --user pybind11
```

## 빌드 하는 법
기본 빌드:
```bash
cd ~/go2_monitor
source /opt/ros/humble/setup.bash
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
colcon build --packages-select go2_monitor_cpp \
  --cmake-args -DCMAKE_PREFIX_PATH="$HOME/.local"
```
주요 산출물:
- 실행 파일: `install/go2_monitor_cpp/lib/go2_monitor_cpp/monitor_server`
- Python 배포 번들 경로: `install/go2_monitor_cpp/lib/go2_monitor_cpp/python_deploy`

## 빌드 옵션 및 실행 옵션
### 빌드 옵션
빌드 타입 예시:
```bash
colcon build --packages-select go2_monitor_cpp \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH="$HOME/.local"
```
```bash
colcon build --packages-select go2_monitor_cpp \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$HOME/.local"
```

추가 CMake 옵션:
| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `-DGO2_MONITOR_BUILD_BENCHMARK=ON` | `OFF` | `viewer_benchmark_server` 추가 빌드 |
| `-DGO2_MONITOR_BUILD_PYTHON_DEPLOYMENT=ON` | `OFF` | 재배포 가능한 Python 번들 생성 |
| `-DGO2_MONITOR_PYTHON_DEPLOY_DIR=...` | `lib/go2_monitor_cpp/python_deploy` | Python 배포 번들 출력 경로 변경 |

옵션 사용 예시:
```bash
colcon build --packages-select go2_monitor_cpp \
  --cmake-args \
    -DCMAKE_PREFIX_PATH="$HOME/.local" \
    -DGO2_MONITOR_BUILD_BENCHMARK=ON
```
```bash
colcon build --packages-select go2_monitor_cpp \
  --cmake-args \
    -DCMAKE_PREFIX_PATH="$HOME/.local" \
    -DGO2_MONITOR_BUILD_PYTHON_DEPLOYMENT=ON
```

### 실행 옵션
코드 기준 CLI 옵션은 아래 네 개다.
| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--source=` | `zenoh` | `zenoh` 또는 로컬 경로. `.db` 파일이면 로그 뷰어로 동작 |
| `--endpoint=` | `udp/127.0.0.1:7447` | Zenoh endpoint. `source=zenoh`일 때 사용 |
| `--keyexpr=` | `**` | Zenoh subscription key expression |
| `--port=` | `8080` | HTTP 서버 포트 |

참고:
- `.db`가 아닌 일반 파일 경로를 주면 파일 내용을 WebSocket 초기 메시지로 전송한다.
- 디렉터리 경로를 주면 디렉터리 목록을 전송한다.
- 뷰어 용도라면 `zenoh` 또는 `.db`를 사용하면 된다.

## 실행하는 법
실행 전 공통 환경:
```bash
cd ~/go2_monitor
source /opt/ros/humble/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
```
브라우저 주소: `http://127.0.0.1:8080`

### 1. 실시간 소스 실행 (`source=zenoh`)
```bash
ros2 run go2_monitor_cpp monitor_server --source=zenoh
ros2 run go2_monitor_cpp monitor_server --source=zenoh --endpoint=udp/192.168.0.151:7447 --port=8080
```

'''
source /opt/ros/humble/setup.bash
cd /home/alice/go2_monitor
source install/setup.bash
ros2 run go2_monitor_cpp monitor_server --source=zenoh

'''
브리지 예시:
```bash
zenoh-bridge-ros2dds -l udp/0.0.0.0:7447
```
커스텀 endpoint / port:
```bash
ros2 run go2_monitor_cpp monitor_server \
  --source=zenoh \
  --endpoint=udp/192.168.0.10:7447 \
  --port=8080
```

### 2. DB 로그 실행 (`source=/path/to/log.db`)
```bash
ros2 run go2_monitor_cpp monitor_server \
  --source=/home/sjs/go2_monitor/log/20260320-152906/log.db \
  --port=8080
```

## 디버깅 모드
gdb 실행:
```bash
ros2 run --prefix 'gdb -ex run --args' go2_monitor_cpp monitor_server --source=zenoh
```
