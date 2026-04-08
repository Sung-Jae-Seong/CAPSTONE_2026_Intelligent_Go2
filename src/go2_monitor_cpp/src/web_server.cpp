#include "go2_monitor_cpp/web_server.hpp"
#include "go2_monitor_cpp/viewer_data_source.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* RosaNavHost = "192.168.0.151";
constexpr int kRemoteInstructionPort = 5000;
constexpr const char* kRemoteInstructionPath = "/query";
constexpr int kConnectTimeoutMs = 2000;
constexpr int kRequestIoTimeoutMs = 100000;

struct InstructionRequestResult {
  bool ok = false;
  int status_code = 0;
  std::string body;
  std::string error;
};

struct SocketHandle {
  int fd = -1;

  ~SocketHandle() {
    if (fd >= 0) {
      ::close(fd);
    }
  }
};

bool write_all(int fd, const std::string& data, std::string& error) {
  std::size_t written = 0;
  while (written < data.size()) {
    const ssize_t rc = ::send(fd, data.data() + written, data.size() - written, 0);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = "failed to write instruction request: " + std::string(std::strerror(errno));
      return false;
    }
    if (rc == 0) {
      error = "failed to write instruction request: connection closed";
      return false;
    }
    written += static_cast<std::size_t>(rc);
  }
  return true;
}

bool connect_with_timeout(int fd, const sockaddr_in& address, std::string& error) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    error = "failed to inspect request socket flags: " + std::string(std::strerror(errno));
    return false;
  }

  if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    error = "failed to configure request socket: " + std::string(std::strerror(errno));
    return false;
  }

  int rc = ::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
  if (rc < 0 && errno == EINPROGRESS) {
    pollfd descriptor {};
    descriptor.fd = fd;
    descriptor.events = POLLOUT;

    while (true) {
      rc = ::poll(&descriptor, 1, kConnectTimeoutMs);
      if (rc < 0 && errno == EINTR) {
        continue;
      }
      break;
    }

    if (rc == 0) {
      error = "failed to connect to request endpoint: timed out";
      return false;
    }
    if (rc < 0) {
      error = "failed to wait for request endpoint: " + std::string(std::strerror(errno));
      return false;
    }

    int socket_error = 0;
    socklen_t socket_error_size = sizeof(socket_error);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &socket_error_size) < 0) {
      error = "failed to inspect request socket: " + std::string(std::strerror(errno));
      return false;
    }
    if (socket_error != 0) {
      error = "failed to connect to request endpoint: " + std::string(std::strerror(socket_error));
      return false;
    }
  } else if (rc < 0) {
    error = "failed to connect to request endpoint: " + std::string(std::strerror(errno));
    return false;
  }

  if (::fcntl(fd, F_SETFL, flags) < 0) {
    error = "failed to restore request socket flags: " + std::string(std::strerror(errno));
    return false;
  }

  timeval timeout {};
  timeout.tv_sec = kRequestIoTimeoutMs / 1000;
  timeout.tv_usec = (kRequestIoTimeoutMs % 1000) * 1000;
  if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0 ||
      ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
    error = "failed to configure request socket timeout: " + std::string(std::strerror(errno));
    return false;
  }

  return true;
}

InstructionRequestResult post_text_request(
  const char* host,
  int port,
  const char* path,
  const std::string& body) {
  InstructionRequestResult result;

  SocketHandle socket_handle;
  socket_handle.fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket_handle.fd < 0) {
    result.error = "failed to open request socket: " + std::string(std::strerror(errno));
    return result;
  }

  sockaddr_in address {};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host, &address.sin_addr) != 1) {
    result.error = "failed to parse request endpoint address";
    return result;
  }

  if (!connect_with_timeout(socket_handle.fd, address, result.error)) {
    return result;
  }

  std::ostringstream request_stream;
  request_stream
    << "POST " << path << " HTTP/1.1\r\n"
    << "Host: " << host << ":" << port << "\r\n"
    << "Content-Type: application/x-www-form-urlencoded\r\n"
    << "Content-Length: " << body.size() << "\r\n"
    << "Connection: close\r\n"
    << "\r\n"
    << body;
  const std::string request = request_stream.str();

  if (!write_all(socket_handle.fd, request, result.error)) {
    return result;
  }

  ::shutdown(socket_handle.fd, SHUT_WR);

  std::array<char, 4096> buffer {};
  std::string raw_response;
  while (true) {
    const ssize_t rc = ::recv(socket_handle.fd, buffer.data(), buffer.size(), 0);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      result.error = "failed to read request response: " + std::string(std::strerror(errno));
      return result;
    }
    if (rc == 0) {
      break;
    }
    raw_response.append(buffer.data(), static_cast<std::size_t>(rc));
  }

  const std::size_t status_line_end = raw_response.find("\r\n");
  const std::size_t header_end = raw_response.find("\r\n\r\n");
  if (status_line_end == std::string::npos || header_end == std::string::npos) {
    result.error = "request endpoint returned an invalid HTTP response";
    return result;
  }

  std::istringstream status_stream(raw_response.substr(0, status_line_end));
  std::string http_version;
  status_stream >> http_version >> result.status_code;
  if (!status_stream || http_version.rfind("HTTP/", 0) != 0) {
    result.error = "request endpoint returned an invalid HTTP status line";
    return result;
  }

  result.body = raw_response.substr(header_end + 4);
  result.ok = result.status_code >= 200 && result.status_code < 300;
  if (!result.ok) {
    if (!result.body.empty()) {
      result.error = result.body;
    } else {
      result.error = "request endpoint returned HTTP " + std::to_string(result.status_code);
    }
  }

  return result;
}

InstructionRequestResult forward_instruction_request(const std::string& instruction) {
  return post_text_request(
    RosaNavHost,
    kRemoteInstructionPort,
    kRemoteInstructionPath,
    instruction);
}

crow::json::wvalue make_unavailable_response(const char* error) {
  crow::json::wvalue response;
  response["ok"] = false;
  response["error"] = error;
  return response;
}

crow::json::wvalue make_source_error_response(const go2_monitor_cpp::ViewerDataSource& viewer_data_source) {
  crow::json::wvalue response;
  response["ok"] = false;
  response["source"] = viewer_data_source.source_label();
  response["source_mode"] = viewer_data_source.source_mode();
  response["error"] = viewer_data_source.last_error();
  return response;
}

crow::json::wvalue::list build_topic_list(const std::vector<go2_monitor_cpp::TopicInfo>& topics) {
  crow::json::wvalue::list topic_list;
  topic_list.reserve(topics.size());

  for (const auto& topic : topics) {
    crow::json::wvalue item;
    item["name"] = topic.name;
    item["type"] = topic.type;
    topic_list.emplace_back(std::move(item));
  }

  return topic_list;
}

std::string make_viewer_image_url(
  std::size_t index,
  const char* stream,
  const go2_monitor_cpp::ViewerFrameInfo& frame,
  const std::string& source_mode) {
  const bool is_rgb = std::strcmp(stream, "rgb") == 0;
  std::string url = "/api/viewer/image?stream=";
  url += stream;
  url += "&index=";
  url += std::to_string(index);
  if (source_mode == "zenoh") {
    const auto revision = is_rgb ? frame.rgb_revision : frame.depth_revision;
    url += "&rev=" + std::to_string(revision);
  }
  return url;
}

crow::json::wvalue::list build_frame_list(const go2_monitor_cpp::ViewerSnapshot& snapshot) {
  crow::json::wvalue::list frame_list;
  frame_list.reserve(snapshot.frames.size());

  for (const auto& frame : snapshot.frames) {
    crow::json::wvalue item;
    item["index"] = static_cast<long long>(frame.index);
    item["rgb_url"] = make_viewer_image_url(frame.index, "rgb", frame, snapshot.source_mode);
    item["depth_url"] = make_viewer_image_url(frame.index, "depth", frame, snapshot.source_mode);
    frame_list.emplace_back(std::move(item));
  }

  return frame_list;
}

crow::json::wvalue::list build_timeline_list(
  const std::vector<go2_monitor_cpp::PlaybackTimelineEntry>& timeline) {
  crow::json::wvalue::list timeline_list;
  timeline_list.reserve(timeline.size());

  for (const auto& entry : timeline) {
    crow::json::wvalue item;
    item["index"] = static_cast<long long>(entry.index);
    item["viewer_frame_index"] = static_cast<long long>(entry.viewer_frame_index);
    item["timestamp"] = entry.timestamp;
    item["has_odom"] = entry.has_odom;
    item["odom_index"] = static_cast<long long>(entry.odom_index);
    item["odom_timestamp"] = entry.odom_timestamp;
    item["odom_x"] = entry.odom_x;
    item["odom_y"] = entry.odom_y;
    item["has_stdout"] = entry.has_stdout;
    item["stdout_index"] = static_cast<long long>(entry.stdout_index);
    timeline_list.emplace_back(std::move(item));
  }

  return timeline_list;
}

crow::json::wvalue::list build_trajectory_list(
  const std::vector<go2_monitor_cpp::TrajectoryPoint>& trajectory) {
  crow::json::wvalue::list trajectory_list;
  trajectory_list.reserve(trajectory.size());

  for (const auto& point : trajectory) {
    crow::json::wvalue item;
    item["timestamp"] = point.timestamp;
    item["x"] = point.x;
    item["y"] = point.y;
    item["has_yaw"] = point.has_yaw;
    item["yaw"] = point.yaw;
    trajectory_list.emplace_back(std::move(item));
  }

  return trajectory_list;
}

crow::json::wvalue::list build_stdout_list(
  const std::vector<go2_monitor_cpp::StdoutEntry>& stdout_entries) {
  crow::json::wvalue::list stdout_list;
  stdout_list.reserve(stdout_entries.size());

  for (const auto& entry : stdout_entries) {
    crow::json::wvalue item;
    item["timestamp"] = entry.timestamp;
    item["message"] = entry.message;
    stdout_list.emplace_back(std::move(item));
  }

  return stdout_list;
}

crow::json::wvalue make_topics_response(
  const go2_monitor_cpp::ViewerDataSource& viewer_data_source,
  const std::vector<go2_monitor_cpp::TopicInfo>& topics) {
  crow::json::wvalue response;
  response["ok"] = true;
  response["source"] = viewer_data_source.source_label();
  response["source_mode"] = viewer_data_source.source_mode();
  response["topics"] = build_topic_list(topics);
  return response;
}

crow::json::wvalue make_viewer_snapshot_response(const go2_monitor_cpp::ViewerSnapshot& snapshot) {
  crow::json::wvalue response;
  response["ok"] = true;
  response["source"] = snapshot.source;
  response["source_mode"] = snapshot.source_mode;
  response["supports_playback"] = snapshot.supports_playback;
  response["revision"] = static_cast<unsigned long long>(snapshot.revision);
  response["rgb_topic"] = snapshot.rgb_topic;
  response["depth_topic"] = snapshot.depth_topic;
  response["stdout_topic"] = snapshot.stdout_topic;
  response["total_frames"] = static_cast<long long>(snapshot.frames.size());
  response["total_timeline_entries"] = static_cast<long long>(snapshot.timeline.size());
  response["frames"] = build_frame_list(snapshot);
  response["timeline"] = build_timeline_list(snapshot.timeline);
  response["trajectory"] = build_trajectory_list(snapshot.trajectory);
  response["stdout_entries"] = build_stdout_list(snapshot.stdout_entries);
  return response;
}

crow::response make_viewer_image_response(
  go2_monitor_cpp::ViewerDataSource& viewer_data_source,
  const crow::request& req) {
  const char* stream = req.url_params.get("stream");
  const char* index_param = req.url_params.get("index");
  if (stream == nullptr || index_param == nullptr) {
    return crow::response(400, "missing stream or index");
  }

  std::size_t index = 0;
  try {
    index = static_cast<std::size_t>(std::stoull(index_param));
  } catch (...) {
    return crow::response(400, "invalid index");
  }

  std::string mime_type;
  std::string image_bytes;
  if (!viewer_data_source.render_viewer_image(index, stream, mime_type, image_bytes)) {
    return crow::response(404, viewer_data_source.last_error());
  }

  crow::response response;
  response.code = 200;
  response.set_header("Content-Type", mime_type);
  response.set_header("Cache-Control", "no-store");
  response.body = std::move(image_bytes);
  return response;
}

crow::json::wvalue make_instruction_response(const InstructionRequestResult& result) {
  crow::json::wvalue response;
  response["ok"] = result.ok;
  response["status_code"] = result.status_code;
  response["response"] = result.body;
  if (!result.ok) {
    response["error"] = result.error;
  }
  return response;
}

}  // namespace

namespace go2_monitor_cpp {

void WebSocketHub::add(crow::websocket::connection* conn) {
  std::lock_guard<std::mutex> lock(mutex_);
  conns_.insert(conn);
}

void WebSocketHub::remove(crow::websocket::connection* conn) {
  std::lock_guard<std::mutex> lock(mutex_);
  conns_.erase(conn);
}

void WebSocketHub::broadcast(const std::string& msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto* conn : conns_) {
    if (conn != nullptr) {
      conn->send_text(msg);
    }
  }
}

WebServer::WebServer(
  std::shared_ptr<WebSocketHub> ws_hub,
  std::shared_ptr<ViewerDataSource> viewer_data_source,
  std::string initial_ws_message)
: ws_hub_(std::move(ws_hub)),
  viewer_data_source_(std::move(viewer_data_source)),
  initial_ws_message_(std::move(initial_ws_message)) {
  setup_routes();
}

void WebServer::run(uint16_t port) {
  app_.port(port).multithreaded().run();
}

void WebServer::stop() {
  app_.stop();
}

void WebServer::setup_routes() {
  CROW_ROUTE(app_, "/")([this]() {
    return index_html();
  });

  CROW_ROUTE(app_, "/api/topics")([this]() {
    if (!viewer_data_source_) {
      return make_unavailable_response(
        "topic api is available only when the current source exposes viewer data");
    }

    std::vector<TopicInfo> topics;
    if (!viewer_data_source_->list_topics(topics)) {
      return make_source_error_response(*viewer_data_source_);
    }

    return make_topics_response(*viewer_data_source_, topics);
  });

  CROW_ROUTE(app_, "/api/viewer/frames")([this]() {
    if (!viewer_data_source_) {
      return make_unavailable_response(
        "viewer api is available only when the current source exposes viewer data");
    }

    ViewerSnapshot snapshot;
    if (!viewer_data_source_->get_viewer_snapshot(snapshot)) {
      return make_source_error_response(*viewer_data_source_);
    }

    return make_viewer_snapshot_response(snapshot);
  });

  CROW_ROUTE(app_, "/api/viewer/image")([this](const crow::request& req) {
    if (!viewer_data_source_) {
      return crow::response(404, "viewer api is available only when the current source exposes viewer data");
    }

    return make_viewer_image_response(*viewer_data_source_, req);
  });

  CROW_ROUTE(app_, "/api/live/instruction")
    .methods(crow::HTTPMethod::POST)([this](const crow::request& req) {
      if (!viewer_data_source_ || viewer_data_source_->source_mode() != "zenoh") {
        return make_unavailable_response("instruction api is available only in zenoh mode");
      }

      if (req.body.empty()) {
        return make_unavailable_response("instruction text is empty");
      }

      return make_instruction_response(forward_instruction_request(req.body));
    });

  CROW_WEBSOCKET_ROUTE(app_, "/ws")
    .onopen([this](crow::websocket::connection& conn) {
      ws_hub_->add(&conn);
      if (!initial_ws_message_.empty()) {
        conn.send_text(initial_ws_message_);
      }
    })
    .onclose([this](crow::websocket::connection& conn, const std::string&) {
      ws_hub_->remove(&conn);
    });
}

std::string WebServer::index_html() const {
  return R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>go2 monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg-top: #fff8ef;
      --bg-bottom: #efe5d8;
      --panel: rgba(255, 255, 255, 0.80);
      --line: #ddd2c3;
      --ink: #1f2933;
      --muted: #667085;
      --accent: #c75c31;
      --accent-soft: #fbe2d5;
      --shadow: 0 18px 36px rgba(31, 41, 51, 0.10);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.95), transparent 34%),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
    }

    main {
      max-width: 1344px;
      margin: 0 auto;
      padding: 22px 12px 28px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-size: clamp(24px, 4vw, 38px);
      line-height: 1;
      letter-spacing: -0.04em;
    }

    .subtitle {
      margin: 12px 0 0;
      max-width: 900px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.6;
    }

    .grid {
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }

    .card {
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.9);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
      padding: 13px;
      min-width: 0;
    }

    .topics-card {
      padding: 14px;
    }

    .section-title {
      margin: 0 0 10px;
      font-size: 20px;
      letter-spacing: -0.03em;
    }

    .topic-list {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 10px;
      min-width: 0;
    }

    .topic-item {
      min-width: 0;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.76);
    }

    .topic-name {
      margin: 0;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }

    .topic-meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      align-items: center;
      min-width: 0;
      flex-wrap: wrap;
    }

    .topic-type {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1 1 120px;
    }

    .empty {
      padding: 14px;
      border-radius: 14px;
      border: 1px dashed var(--line);
      color: var(--muted);
      background: rgba(255, 255, 255, 0.56);
      font-size: 13px;
    }

    .viewer-controls {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
      padding: 0;
      border: none;
      background: transparent;
    }

    .viewer-controls[hidden] {
      display: none !important;
    }

    .button-row {
      display: flex;
      flex-wrap: nowrap;
      gap: 8px;
      flex: 0 0 auto;
    }

    .control-button {
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 9px 13px;
      background: #1f2933;
      color: #fff8ef;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.16s ease, opacity 0.16s ease, background 0.16s ease;
    }

    .control-button:hover {
      transform: translateY(-1px);
    }

    .control-button.secondary {
      background: #d9cec0;
      color: #3d4752;
    }

    .control-button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
    }

    .timeline-row {
      display: grid;
      gap: 6px;
      flex: 1 1 auto;
      min-width: 0;
    }

    .timeline-meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
    }

    .timeline-meta strong {
      white-space: nowrap;
      flex: 0 0 auto;
    }

    .timeline-meta span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-align: right;
      flex: 1 1 auto;
    }

    .timeline-row input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .chat-composer {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      min-width: 0;
    }

    .chat-input-field {
      width: 100%;
      min-width: 0;
      min-height: 44px;
      max-height: 44px;
      resize: none;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 11px 14px;
      background: rgba(255, 255, 255, 0.82);
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      line-height: 1.4;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
    }

    .chat-input-field::placeholder {
      color: var(--muted);
    }

    .chat-input-field:focus {
      outline: none;
      border-color: rgba(199, 92, 49, 0.55);
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.55),
        0 0 0 3px rgba(199, 92, 49, 0.14);
    }

    .chat-submit-button {
      min-width: 44px;
      width: 44px;
      height: 44px;
      padding: 0;
      font-size: 18px;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
    }

    .viewer-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    .viewer-grid.single-panel {
      grid-template-columns: 1fr;
    }

    .support-grid {
      margin-top: 14px;
    }

    .viewer-card {
      padding: 0;
      border: none;
      background: transparent;
    }

    .viewer-title {
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .viewer-frame {
      width: 100%;
      aspect-ratio: 4 / 3;
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      background: #f4ede5;
      border: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .viewer-frame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #f8f5f1;
    }

    .viewer-overlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }

    .pixel-goal-marker {
      position: absolute;
      display: none;
      transform: translate(-50%, -50%);
    }

    .pixel-goal-marker.visible {
      display: block;
    }

    .pixel-goal-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #e11d48;
      border: 2px solid #fff8ef;
      box-shadow: 0 0 0 2px rgba(225, 29, 72, 0.20);
      display: block;
    }

    .pixel-goal-label {
      position: absolute;
      left: 14px;
      top: -9px;
      border-radius: 999px;
      padding: 3px 8px;
      background: rgba(31, 41, 51, 0.88);
      color: #fff8ef;
      font-size: 11px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }

    .stdout-readout {
      color: var(--muted);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      text-align: right;
      word-break: break-word;
    }

    .stdout-display {
      margin: 0;
      width: 100%;
      height: 256px;
      padding: 12px 14px;
      overflow: auto;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #f8f5f1;
      color: var(--ink);
    }

    .chat-thread {
      display: grid;
      gap: 10px;
      align-content: start;
    }

    .chat-row {
      display: flex;
      width: 100%;
    }

    .chat-row.user {
      justify-content: flex-end;
    }

    .chat-row.assistant,
    .chat-row.system {
      justify-content: flex-start;
    }

    .chat-bubble {
      max-width: min(78%, 560px);
      padding: 10px 12px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 8px 18px rgba(31, 41, 51, 0.06);
      min-width: 0;
    }

    .chat-row.user .chat-bubble {
      background: linear-gradient(180deg, #1f2933, #334155);
      border-color: rgba(31, 41, 51, 0.12);
      color: #fff8ef;
    }

    .chat-row.system .chat-bubble {
      background: rgba(217, 206, 192, 0.42);
    }

    .chat-role {
      margin: 0 0 4px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .chat-row.user .chat-role {
      color: rgba(255, 248, 239, 0.78);
    }

    .chat-message {
      margin: 0;
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .trajectory-panel {
      margin-top: 0;
      padding: 0;
      border: none;
      background: transparent;
    }

    .trajectory-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }

    .pose-readout {
      color: var(--muted);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      text-align: right;
      word-break: break-word;
    }

    .trajectory-note {
      position: absolute;
      left: 12px;
      bottom: 10px;
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
      background: rgba(248, 245, 241, 0.92);
      padding: 2px 6px;
      border-radius: 8px;
      pointer-events: none;
    }

    .trajectory-stage {
      position: relative;
    }

    #trajectory-canvas {
      width: 100%;
      height: 256px;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #f8f5f1;
    }

    @media (max-width: 900px) {
      .hero {
        flex-direction: column;
      }

      .grid {
        grid-template-columns: 1fr;
      }

      .viewer-controls {
        flex-direction: column;
        align-items: stretch;
      }

      .chat-composer {
        flex-direction: column;
        align-items: stretch;
      }

      .button-row {
        flex-wrap: wrap;
      }

      .chat-submit-button {
        width: 100%;
      }

      .viewer-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>go2 monitor</h1>
    </section>

    <section class="grid">
      <section class="card topics-card">
        <h2 class="section-title">Topics</h2>
        <ul class="topic-list" id="topics">
          <li class="empty">Loading topics...</li>
        </ul>
      </section>

      <section class="card">
        <h2 class="section-title">Viewers</h2>

        <section class="viewer-controls" id="viewer-controls" hidden>
          <div class="button-row">
            <button class="control-button secondary" id="btn-prev" type="button">Prev</button>
            <button class="control-button" id="btn-play" type="button">Play</button>
            <button class="control-button secondary" id="btn-next" type="button">Next</button>
          </div>

          <div class="timeline-row">
            <div class="timeline-meta">
              <strong id="frame-label">No frames</strong>
              <span id="frame-time">viewer unavailable</span>
            </div>
            <input id="frame-slider" type="range" min="0" max="0" value="0" step="1" disabled>
          </div>
        </section>

        <div class="viewer-grid" id="viewer-grid">
          <section class="viewer-card">
            <p class="viewer-title">RGB</p>
            <div class="viewer-frame">
              <img
                id="viewer-left"
                alt="RGB viewer"
                src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='800' height='600'><rect width='100%' height='100%' fill='%23f4ede5'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-family='Segoe UI, sans-serif' font-size='28' fill='%23667085'>No image</text></svg>"
              >
              <div class="viewer-overlay">
                <div class="pixel-goal-marker" id="pixel-goal-marker">
                  <span class="pixel-goal-dot"></span>
                  <span class="pixel-goal-label">pixel goal</span>
                </div>
              </div>
            </div>
          </section>

          <section class="viewer-card" id="depth-card">
            <p class="viewer-title">Depth</p>
            <div class="viewer-frame">
              <img
                id="viewer-right"
                alt="Depth viewer"
                src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='800' height='600'><rect width='100%' height='100%' fill='%23f4ede5'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-family='Segoe UI, sans-serif' font-size='28' fill='%23667085'>No image</text></svg>"
              >
            </div>
          </section>
        </div>

        <div class="viewer-grid support-grid" id="support-grid">
          <section class="viewer-card" id="stdout-card">
            <div class="panel-head">
              <p class="viewer-title">Print / Stdout</p>
              <div class="stdout-readout" id="stdout-readout">stdout unavailable</div>
            </div>
            <div class="stdout-display" id="stdout-viewer">
              <div class="chat-thread" id="stdout-thread">
                <div class="chat-row system">
                  <div class="chat-bubble">
                    <p class="chat-role">Stdout</p>
                    <p class="chat-message">No stdout data</p>
                  </div>
                </div>
              </div>
            </div>
            <section class="chat-composer" id="live-input-bar" hidden>
              <textarea
                class="chat-input-field"
                id="live-input-field"
                rows="1"
                placeholder="Type here..."
                aria-label="Live input"
              ></textarea>
              <button
                class="control-button chat-submit-button"
                id="live-submit-button"
                type="button"
                aria-label="Submit"
              >↵</button>
            </section>
          </section>

          <section class="viewer-card trajectory-panel" id="trajectory-card">
            <div class="trajectory-head">
              <p class="viewer-title">Trajectory</p>
              <div class="pose-readout" id="pose-readout">odometry unavailable</div>
            </div>
            <div class="trajectory-stage">
              <canvas id="trajectory-canvas"></canvas>
              <p class="trajectory-note">초기 경로는 정확하지 않을 수 있음</p>
            </div>
          </section>
        </div>
      </section>
    </section>
  </main>

  <script>
    const topicsEl = document.getElementById('topics');
    const viewerLeftEl = document.getElementById('viewer-left');
    const viewerRightEl = document.getElementById('viewer-right');
    const pixelGoalMarkerEl = document.getElementById('pixel-goal-marker');
    const viewerGridEl = document.getElementById('viewer-grid');
    const supportGridEl = document.getElementById('support-grid');
    const depthCardEl = document.getElementById('depth-card');
    const stdoutCardEl = document.getElementById('stdout-card');
    const trajectoryCardEl = document.getElementById('trajectory-card');
    const viewerControlsEl = document.getElementById('viewer-controls');
    const frameSliderEl = document.getElementById('frame-slider');
    const frameLabelEl = document.getElementById('frame-label');
    const frameTimeEl = document.getElementById('frame-time');
    const playButtonEl = document.getElementById('btn-play');
    const prevButtonEl = document.getElementById('btn-prev');
    const nextButtonEl = document.getElementById('btn-next');
    const liveInputBarEl = document.getElementById('live-input-bar');
    const liveInputFieldEl = document.getElementById('live-input-field');
    const liveSubmitButtonEl = document.getElementById('live-submit-button');
    const stdoutViewerEl = document.getElementById('stdout-viewer');
    const stdoutThreadEl = document.getElementById('stdout-thread');
    const stdoutReadoutEl = document.getElementById('stdout-readout');
    const trajectoryCanvasEl = document.getElementById('trajectory-canvas');
    const poseReadoutEl = document.getElementById('pose-readout');

    const viewerState = {
      frames: [],
      timeline: [],
      trajectory: [],
      stdoutEntries: [],
      currentPixelGoal: null,
      currentPlannedTrajectory: null,
      currentIndex: 0,
      timerId: null,
      supportsPlayback: false,
      sourceMode: '',
      liveRevision: -1,
      liveSocket: null,
      liveRefreshTimerId: null,
      liveRefreshInFlight: false,
      liveRefreshPending: false,
      rgbTopic: '/camera/color/image_raw',
      depthTopic: '/camera/aligned_depth_to_color/image_raw',
      stdoutTopic: 'print',
      liveCommandInFlight: false,
      localStdoutEntries: [],
    };

    function updateImageSrc(imageEl, nextUrl) {
      if (!nextUrl) {
        return;
      }

      if (imageEl.dataset.viewerSrc === nextUrl) {
        return;
      }

      imageEl.dataset.viewerSrc = nextUrl;
      imageEl.src = nextUrl;
    }

    function stopPlayback() {
      if (viewerState.timerId !== null) {
        clearInterval(viewerState.timerId);
        viewerState.timerId = null;
      }
      updateControls();
    }

    function updateControls() {
      const hasTimeline = viewerState.timeline.length > 0;
      const playbackEnabled = viewerState.supportsPlayback && hasTimeline;
      const atStart = viewerState.currentIndex <= 0;
      const atEnd = viewerState.currentIndex >= viewerState.timeline.length - 1;
      const isPlaying = viewerState.timerId !== null;
      const showPlaybackControls = viewerState.supportsPlayback;
      const showZenohStdoutLayout = !viewerState.supportsPlayback && viewerState.sourceMode === 'zenoh';

      depthCardEl.hidden = showZenohStdoutLayout;
      if (showZenohStdoutLayout) {
        supportGridEl.classList.add('single-panel');
        if (stdoutCardEl.parentElement !== viewerGridEl) {
          viewerGridEl.appendChild(stdoutCardEl);
        }
      } else {
        supportGridEl.classList.remove('single-panel');
        if (stdoutCardEl.parentElement !== supportGridEl) {
          supportGridEl.insertBefore(stdoutCardEl, trajectoryCardEl);
        }
      }

      viewerControlsEl.hidden = !showPlaybackControls;
      frameSliderEl.disabled = !playbackEnabled;
      playButtonEl.disabled = !playbackEnabled;
      playButtonEl.textContent = isPlaying ? 'Stop' : 'Play';
      prevButtonEl.disabled = !playbackEnabled || atStart;
      nextButtonEl.disabled = !playbackEnabled || atEnd;
      liveInputBarEl.hidden = viewerState.sourceMode !== 'zenoh';
      liveInputFieldEl.disabled = viewerState.sourceMode !== 'zenoh' || viewerState.liveCommandInFlight;
      liveSubmitButtonEl.disabled = viewerState.sourceMode !== 'zenoh' || viewerState.liveCommandInFlight;
    }

    function drawTrajectory() {
      const context = trajectoryCanvasEl.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const cssWidth = Math.max(1, Math.floor(trajectoryCanvasEl.clientWidth));
      const cssHeight = Math.max(1, Math.floor(trajectoryCanvasEl.clientHeight));
      const pixelWidth = Math.floor(cssWidth * dpr);
      const pixelHeight = Math.floor(cssHeight * dpr);

      if (trajectoryCanvasEl.width !== pixelWidth || trajectoryCanvasEl.height !== pixelHeight) {
        trajectoryCanvasEl.width = pixelWidth;
        trajectoryCanvasEl.height = pixelHeight;
      }

      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, cssWidth, cssHeight);

      const points = viewerState.trajectory;
      if (!points.length) {
        poseReadoutEl.textContent = 'odometry unavailable';

        context.fillStyle = '#667085';
        context.font = '14px Segoe UI, sans-serif';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText('No trajectory data', cssWidth / 2, cssHeight / 2);
        return;
      }

      const timelineEntry = viewerState.timeline[viewerState.currentIndex];
      const currentPose = timelineEntry && timelineEntry.has_odom
        ? viewerState.trajectory[Math.min(timelineEntry.odom_index, points.length - 1)]
        : null;
      const worldPlan = [];

      if (
        currentPose &&
        currentPose.has_yaw &&
        Array.isArray(viewerState.currentPlannedTrajectory) &&
        viewerState.currentPlannedTrajectory.length > 0
      ) {
        const cosYaw = Math.cos(currentPose.yaw);
        const sinYaw = Math.sin(currentPose.yaw);
        for (const localPoint of viewerState.currentPlannedTrajectory) {
          worldPlan.push({
            x: currentPose.x + cosYaw * localPoint.x - sinYaw * localPoint.y,
            y: currentPose.y + sinYaw * localPoint.x + cosYaw * localPoint.y,
          });
        }
      }

      let minX = Number.POSITIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;

      for (const point of points) {
        minX = Math.min(minX, point.x);
        minY = Math.min(minY, point.y);
        maxX = Math.max(maxX, point.x);
        maxY = Math.max(maxY, point.y);
      }

      for (const point of worldPlan) {
        minX = Math.min(minX, point.x);
        minY = Math.min(minY, point.y);
        maxX = Math.max(maxX, point.x);
        maxY = Math.max(maxY, point.y);
      }

      const padding = 24;
      const spanX = Math.max(maxX - minX, 1e-6);
      const spanY = Math.max(maxY - minY, 1e-6);
      const scale = Math.min(
        (cssWidth - padding * 2) / spanX,
        (cssHeight - padding * 2) / spanY
      );
      const offsetX = (cssWidth - spanX * scale) / 2;
      const offsetY = (cssHeight - spanY * scale) / 2;

      function project(point) {
        return {
          x: offsetX + (point.x - minX) * scale,
          y: cssHeight - (offsetY + (point.y - minY) * scale),
        };
      }

      context.strokeStyle = '#d9cec0';
      context.lineWidth = 1;
      context.strokeRect(0.5, 0.5, cssWidth - 1, cssHeight - 1);

      context.beginPath();
      points.forEach((point, index) => {
        const projected = project(point);
        if (index === 0) {
          context.moveTo(projected.x, projected.y);
        } else {
          context.lineTo(projected.x, projected.y);
        }
      });
      context.strokeStyle = '#c75c31';
      context.lineWidth = 2;
      context.stroke();

      if (!timelineEntry || !timelineEntry.has_odom) {
        poseReadoutEl.textContent = 'odometry unavailable';
        return;
      }

      const point = viewerState.trajectory[Math.min(timelineEntry.odom_index, points.length - 1)];
      const projected = project(point);
      const hasHeading = Boolean(point && point.has_yaw);

      if (worldPlan.length > 1) {
        context.beginPath();
        worldPlan.forEach((planPoint, index) => {
          const projectedPlanPoint = project(planPoint);
          if (index === 0) {
            context.moveTo(projectedPlanPoint.x, projectedPlanPoint.y);
          } else {
            context.lineTo(projectedPlanPoint.x, projectedPlanPoint.y);
          }
        });
        context.strokeStyle = '#2563eb';
        context.lineWidth = 3;
        context.setLineDash([8, 6]);
        context.stroke();
        context.setLineDash([]);

        const planEnd = project(worldPlan[worldPlan.length - 1]);
        context.beginPath();
        context.arc(planEnd.x, planEnd.y, 4, 0, Math.PI * 2);
        context.fillStyle = '#2563eb';
        context.fill();
      }

      context.beginPath();
      context.arc(projected.x, projected.y, 6, 0, Math.PI * 2);
      context.fillStyle = '#1f2933';
      context.fill();
      context.lineWidth = 2;
      context.strokeStyle = '#fff8ef';
      context.stroke();

      if (hasHeading) {
        const headingLength = 20;
        const tipX = projected.x + Math.cos(point.yaw) * headingLength;
        const tipY = projected.y - Math.sin(point.yaw) * headingLength;
        const wingAngle = 0.5;
        const wingLength = 7;

        context.beginPath();
        context.moveTo(projected.x, projected.y);
        context.lineTo(tipX, tipY);
        context.strokeStyle = '#1f2933';
        context.lineWidth = 3;
        context.stroke();

        context.beginPath();
        context.moveTo(tipX, tipY);
        context.lineTo(
          tipX - Math.cos(point.yaw - wingAngle) * wingLength,
          tipY + Math.sin(point.yaw - wingAngle) * wingLength
        );
        context.lineTo(
          tipX - Math.cos(point.yaw + wingAngle) * wingLength,
          tipY + Math.sin(point.yaw + wingAngle) * wingLength
        );
        context.closePath();
        context.fillStyle = '#1f2933';
        context.fill();
      }

      const headingText = hasHeading
        ? ` · yaw ${(point.yaw * 180 / Math.PI).toFixed(1)}°`
        : '';
      poseReadoutEl.textContent =
        `x ${timelineEntry.odom_x.toFixed(3)} · y ${timelineEntry.odom_y.toFixed(3)}${headingText} · ${timelineEntry.odom_timestamp || 'no stamp'}`;
    }

    function appendLocalStdout(role, message) {
      const speaker = typeof role === 'string' && role.length > 0 ? role : 'GO2';
      const text = typeof message === 'string' ? message : String(message ?? '');
      viewerState.localStdoutEntries.push({
        timestamp: new Date().toISOString(),
        role: speaker,
        message: text,
      });
    }

    function normalizeStdoutEntry(entry) {
      const rawMessage = entry && typeof entry.message === 'string' ? entry.message : '';
      const explicitRole = entry && typeof entry.role === 'string' ? entry.role.trim() : '';
      let role = explicitRole;
      let message = rawMessage;

      if (!role) {
        const prefixMatch = rawMessage.match(/^\s*([^:\n]+)\s*:\s*([\s\S]*)$/);
        if (prefixMatch) {
          role = prefixMatch[1].trim();
          message = prefixMatch[2];
        }
      }

      const normalizedRole = role || 'Stdout';
      const lowerRole = normalizedRole.toLowerCase();
      let side = 'system';
      if (lowerRole === 'user') {
        side = 'user';
      } else if (
        lowerRole === 'go2' ||
        lowerRole === 'assistant' ||
        lowerRole === 'llm' ||
        lowerRole === 'planner'
      ) {
        side = 'assistant';
      }

      return {
        timestamp: entry && entry.timestamp ? entry.timestamp : '',
        role: normalizedRole,
        message: typeof message === 'string' && message.length > 0 ? message : '(empty response)',
        side,
      };
    }

    function renderChatEntries(entries, emptyMessage) {
      stdoutThreadEl.innerHTML = '';

      if (!entries.length) {
        const row = document.createElement('div');
        row.className = 'chat-row system';

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';

        const role = document.createElement('p');
        role.className = 'chat-role';
        role.textContent = 'Stdout';

        const message = document.createElement('p');
        message.className = 'chat-message';
        message.textContent = emptyMessage;

        bubble.append(role, message);
        row.append(bubble);
        stdoutThreadEl.append(row);
        return;
      }

      for (const entry of entries) {
        const normalized = normalizeStdoutEntry(entry);
        const row = document.createElement('div');
        row.className = `chat-row ${normalized.side}`;

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';

        const role = document.createElement('p');
        role.className = 'chat-role';
        role.textContent = normalized.role;

        const message = document.createElement('p');
        message.className = 'chat-message';
        message.textContent = normalized.message;

        bubble.append(role, message);
        row.append(bubble);
        stdoutThreadEl.append(row);
      }
    }

    function extractPixelGoal(message) {
      if (typeof message !== 'string' || message.length === 0) {
        return null;
      }

      const match = message.match(/"pixel_goal"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]/);
      if (!match) {
        return null;
      }

      const x = Number(match[1]);
      const y = Number(match[2]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return null;
      }

      return { x, y };
    }

    function extractPlannedTrajectory(message) {
      if (typeof message !== 'string' || message.length === 0) {
        return null;
      }

      const jsonStart = message.indexOf('{');
      if (jsonStart < 0) {
        return null;
      }

      try {
        const payload = JSON.parse(message.slice(jsonStart));
        if (!payload || !Array.isArray(payload.trajectory)) {
          return null;
        }

        const points = [];
        for (const item of payload.trajectory) {
          if (!Array.isArray(item) || item.length < 2) {
            continue;
          }

          const x = Number(item[0]);
          const y = Number(item[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            continue;
          }

          points.push({ x, y });
        }

        return points.length > 0 ? points : null;
      } catch (error) {
        return null;
      }
    }

    function updateCurrentFrameAnnotations() {
      if (!viewerState.timeline.length || !viewerState.stdoutEntries.length) {
        viewerState.currentPixelGoal = null;
        viewerState.currentPlannedTrajectory = null;
        return;
      }

      const timelineEntry = viewerState.timeline[viewerState.currentIndex];
      if (!timelineEntry || !timelineEntry.has_stdout) {
        viewerState.currentPixelGoal = null;
        viewerState.currentPlannedTrajectory = null;
        return;
      }

      const prevTimelineEntry = viewerState.currentIndex > 0
        ? viewerState.timeline[viewerState.currentIndex - 1]
        : null;
      const startIndex = prevTimelineEntry && prevTimelineEntry.has_stdout
        ? prevTimelineEntry.stdout_index + 1
        : 0;
      const endIndex = Math.min(timelineEntry.stdout_index, viewerState.stdoutEntries.length - 1);

      let frameGoal = null;
      let frameTrajectory = null;
      for (let i = startIndex; i <= endIndex; i += 1) {
        const entry = viewerState.stdoutEntries[i];
        const goal = extractPixelGoal(entry && entry.message);
        if (goal) {
          frameGoal = {
            x: goal.x,
            y: goal.y,
            timestamp: entry.timestamp || '',
          };
        }

        const trajectory = extractPlannedTrajectory(entry && entry.message);
        if (trajectory) {
          frameTrajectory = trajectory;
        }
      }

      viewerState.currentPixelGoal = frameGoal;
      viewerState.currentPlannedTrajectory = frameTrajectory;
    }

    function renderPixelGoalOverlay() {
      const goal = viewerState.currentPixelGoal;
      const naturalWidth = viewerLeftEl.naturalWidth;
      const naturalHeight = viewerLeftEl.naturalHeight;
      const boxWidth = viewerLeftEl.clientWidth;
      const boxHeight = viewerLeftEl.clientHeight;

      if (
        !goal ||
        !Number.isFinite(goal.x) ||
        !Number.isFinite(goal.y) ||
        naturalWidth <= 0 ||
        naturalHeight <= 0 ||
        boxWidth <= 0 ||
        boxHeight <= 0
      ) {
        pixelGoalMarkerEl.classList.remove('visible');
        return;
      }

      const imageAspect = naturalWidth / naturalHeight;
      const boxAspect = boxWidth / boxHeight;
      let renderedWidth = 0;
      let renderedHeight = 0;
      let offsetX = 0;
      let offsetY = 0;

      if (imageAspect > boxAspect) {
        renderedWidth = boxWidth;
        renderedHeight = boxWidth / imageAspect;
        offsetY = (boxHeight - renderedHeight) / 2;
      } else {
        renderedHeight = boxHeight;
        renderedWidth = boxHeight * imageAspect;
        offsetX = (boxWidth - renderedWidth) / 2;
      }

      const x = offsetX + (goal.x / naturalWidth) * renderedWidth;
      const y = offsetY + (goal.y / naturalHeight) * renderedHeight;

      if (x < offsetX || x > offsetX + renderedWidth || y < offsetY || y > offsetY + renderedHeight) {
        pixelGoalMarkerEl.classList.remove('visible');
        return;
      }

      pixelGoalMarkerEl.style.left = `${x}px`;
      pixelGoalMarkerEl.style.top = `${y}px`;
      pixelGoalMarkerEl.classList.add('visible');
    }

    function renderStdout() {
      const lastLocalEntry = viewerState.localStdoutEntries.length > 0
        ? viewerState.localStdoutEntries[viewerState.localStdoutEntries.length - 1]
        : null;

      if (!viewerState.timeline.length) {
        if (!viewerState.localStdoutEntries.length) {
          stdoutReadoutEl.textContent = 'stdout unavailable';
          renderChatEntries([], 'No stdout data');
          return;
        }

        stdoutReadoutEl.textContent = lastLocalEntry && lastLocalEntry.timestamp
          ? lastLocalEntry.timestamp
          : 'instruction response';
        renderChatEntries(viewerState.localStdoutEntries, 'No stdout data');
        stdoutViewerEl.scrollTop = stdoutViewerEl.scrollHeight;
        return;
      }

      const timelineEntry = viewerState.timeline[viewerState.currentIndex];
      let baseStdoutEntries = [];
      let baseReadoutText = 'stdout unavailable';

      if (timelineEntry && timelineEntry.has_stdout && viewerState.stdoutEntries.length) {
        const stdoutIndex = Math.min(timelineEntry.stdout_index, viewerState.stdoutEntries.length - 1);
        const stdoutEntry = viewerState.stdoutEntries[stdoutIndex];
        baseReadoutText = stdoutEntry && stdoutEntry.timestamp ? stdoutEntry.timestamp : 'stdout';
        baseStdoutEntries = viewerState.stdoutEntries.slice(0, stdoutIndex + 1);
      }

      const mergedStdoutEntries = baseStdoutEntries.concat(viewerState.localStdoutEntries);
      if (!mergedStdoutEntries.length) {
        stdoutReadoutEl.textContent = 'stdout unavailable';
        renderChatEntries([], 'No stdout data for this time');
        return;
      }

      stdoutReadoutEl.textContent = lastLocalEntry && lastLocalEntry.timestamp
        ? lastLocalEntry.timestamp
        : baseReadoutText;
      renderChatEntries(mergedStdoutEntries, 'No stdout data for this time');
      stdoutViewerEl.scrollTop = stdoutViewerEl.scrollHeight;
    }

    async function submitLiveInstruction() {
      if (viewerState.sourceMode !== 'zenoh' || viewerState.liveCommandInFlight) {
        return;
      }

      const instruction = liveInputFieldEl.value.trim();
      if (!instruction) {
        return;
      }

      liveInputFieldEl.value = '';
      appendLocalStdout('User', instruction);
      renderStdout();
      viewerState.liveCommandInFlight = true;
      updateControls();

      try {
        const response = await fetch('/api/live/instruction', {
          method: 'POST',
          headers: {
            'Content-Type': 'text/plain;charset=UTF-8',
          },
          body: instruction,
        });

        const payload = await response.json();
        if (payload && payload.ok) {
          appendLocalStdout('GO2', typeof payload.response === 'string' && payload.response.length > 0
            ? payload.response
            : '(empty response)');
        } else {
          appendLocalStdout(
            'GO2',
            payload && typeof payload.error === 'string' && payload.error.length > 0
              ? payload.error
              : 'Failed to update instruction');
        }
      } catch (error) {
        appendLocalStdout(
          'GO2',
          error && typeof error.message === 'string' && error.message.length > 0
            ? `Failed to update instruction: ${error.message}`
            : 'Failed to update instruction');
      } finally {
        viewerState.liveCommandInFlight = false;
        renderStdout();
        updateControls();
      }
    }

    function renderFrame() {
      if (!viewerState.timeline.length) {
        if (viewerState.sourceMode === 'zenoh') {
          frameLabelEl.textContent = 'Waiting for live frames';
          frameTimeEl.textContent = 'zenoh stream connected';
        } else {
          frameLabelEl.textContent = 'No frames';
          frameTimeEl.textContent = 'viewer unavailable';
        }
        frameSliderEl.value = '0';
        frameSliderEl.max = '0';
        renderStdout();
        updateCurrentFrameAnnotations();
        renderPixelGoalOverlay();
        drawTrajectory();
        updateControls();
        return;
      }

      const timelineEntry = viewerState.timeline[viewerState.currentIndex];
      const frame = viewerState.frames[timelineEntry.viewer_frame_index];
      if (!frame) {
        frameLabelEl.textContent = 'No frames';
        frameTimeEl.textContent = 'viewer unavailable';
        renderStdout();
        updateCurrentFrameAnnotations();
        renderPixelGoalOverlay();
        drawTrajectory();
        updateControls();
        return;
      }

      updateImageSrc(viewerLeftEl, frame.rgb_url);
      if (viewerState.sourceMode !== 'zenoh' || viewerState.supportsPlayback) {
        updateImageSrc(viewerRightEl, frame.depth_url);
      }
      frameSliderEl.max = String(viewerState.timeline.length - 1);
      frameSliderEl.value = String(viewerState.currentIndex);
      frameLabelEl.textContent = viewerState.supportsPlayback
        ? `Frame ${viewerState.currentIndex + 1} / ${viewerState.timeline.length}`
        : 'Live frame';
      frameTimeEl.textContent = timelineEntry.timestamp || 'timestamp unavailable';
      renderStdout();
      updateCurrentFrameAnnotations();
      renderPixelGoalOverlay();
      drawTrajectory();
      updateControls();
    }

    function renderTopics(topics) {
      if (!topics.length) {
        topicsEl.innerHTML = '<li class="empty">No topics found in the current database.</li>';
        return;
      }

      topicsEl.innerHTML = '';

      for (const topic of topics) {
        const item = document.createElement('li');
        item.className = 'topic-item';

        const name = document.createElement('p');
        name.className = 'topic-name';
        name.textContent = topic.name;

        const meta = document.createElement('div');
        meta.className = 'topic-meta';

        const type = document.createElement('span');
        type.className = 'topic-type';
        type.textContent = topic.type;
        meta.append(type);

        item.append(name, meta);
        topicsEl.append(item);
      }
    }

    async function loadTopics() {
      try {
        const response = await fetch('/api/topics');
        const payload = await response.json();

        if (!payload.ok) {
          topicsEl.innerHTML = `<li class="empty">${payload.error || 'Topic data unavailable.'}</li>`;
          return;
        }

        renderTopics(payload.topics || []);
      } catch (error) {
        topicsEl.innerHTML = '<li class="empty">Failed to load /api/topics.</li>';
      }
    }

    async function loadViewerFrames() {
      try {
        const response = await fetch('/api/viewer/frames');
        const payload = await response.json();

        if (!payload.ok) {
          viewerState.sourceMode = payload.source_mode || viewerState.sourceMode;
          frameLabelEl.textContent = 'No frames';
          frameTimeEl.textContent = payload.error || 'viewer unavailable';
          updateControls();
          return;
        }

        viewerState.supportsPlayback = Boolean(payload.supports_playback);
        viewerState.sourceMode = payload.source_mode || '';
        viewerState.liveRevision = Number(payload.revision) || 0;
        viewerState.rgbTopic = payload.rgb_topic || viewerState.rgbTopic;
        viewerState.depthTopic = payload.depth_topic || viewerState.depthTopic;
        viewerState.stdoutTopic = payload.stdout_topic || viewerState.stdoutTopic;
        viewerState.frames = payload.frames || [];
        viewerState.timeline = payload.timeline || [];
        viewerState.trajectory = payload.trajectory || [];
        viewerState.stdoutEntries = payload.stdout_entries || [];
        viewerState.currentIndex = 0;
        renderFrame();
      } catch (error) {
        frameLabelEl.textContent = 'No frames';
        frameTimeEl.textContent = 'Failed to load viewer frames';
        drawTrajectory();
        updateControls();
      }
    }

    function scheduleLiveRefresh() {
      if (viewerState.supportsPlayback || viewerState.sourceMode !== 'zenoh') {
        return;
      }

      if (viewerState.liveRefreshInFlight) {
        viewerState.liveRefreshPending = true;
        return;
      }

      if (viewerState.liveRefreshTimerId !== null) {
        return;
      }

      viewerState.liveRefreshTimerId = window.setTimeout(async () => {
        viewerState.liveRefreshTimerId = null;
        viewerState.liveRefreshInFlight = true;

        try {
          await loadViewerFrames();
        } finally {
          viewerState.liveRefreshInFlight = false;
          if (viewerState.liveRefreshPending) {
            viewerState.liveRefreshPending = false;
            scheduleLiveRefresh();
          }
        }
      }, 80);
    }

    function connectLiveViewer() {
      if (viewerState.supportsPlayback || viewerState.sourceMode !== 'zenoh' || viewerState.liveSocket !== null) {
        return;
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
      viewerState.liveSocket = socket;

      socket.addEventListener('message', (event) => {
        let payload = null;
        try {
          payload = JSON.parse(event.data);
        } catch (error) {
          return;
        }

        if (!payload || payload.kind !== 'viewer_update') {
          return;
        }

        const revision = Number(payload.revision);
        if (Number.isFinite(revision) && revision === viewerState.liveRevision) {
          return;
        }

        scheduleLiveRefresh();
      });

      socket.addEventListener('close', () => {
        if (viewerState.liveSocket === socket) {
          viewerState.liveSocket = null;
        }

        if (!viewerState.supportsPlayback && viewerState.sourceMode === 'zenoh') {
          window.setTimeout(connectLiveViewer, 1000);
        }
      });
    }

    playButtonEl.addEventListener('click', () => {
      if (!viewerState.timeline.length) {
        return;
      }

      if (viewerState.timerId !== null) {
        stopPlayback();
        return;
      }

      if (viewerState.currentIndex >= viewerState.timeline.length - 1) {
        viewerState.currentIndex = 0;
        renderFrame();
      }

      stopPlayback();
      viewerState.timerId = window.setInterval(() => {
        if (viewerState.currentIndex >= viewerState.timeline.length - 1) {
          stopPlayback();
          return;
        }

        viewerState.currentIndex += 1;
        renderFrame();
      }, 300);
      updateControls();
    });

    prevButtonEl.addEventListener('click', () => {
      if (viewerState.currentIndex <= 0) {
        return;
      }

      stopPlayback();
      viewerState.currentIndex -= 1;
      renderFrame();
    });

    nextButtonEl.addEventListener('click', () => {
      if (viewerState.currentIndex >= viewerState.timeline.length - 1) {
        return;
      }

      stopPlayback();
      viewerState.currentIndex += 1;
      renderFrame();
    });

    frameSliderEl.addEventListener('input', (event) => {
      stopPlayback();
      viewerState.currentIndex = Number(event.target.value) || 0;
      renderFrame();
    });

    liveSubmitButtonEl.addEventListener('click', submitLiveInstruction);
    liveInputFieldEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        submitLiveInstruction();
      }
    });

    loadTopics();
    loadViewerFrames().then(connectLiveViewer);
    viewerLeftEl.addEventListener('load', renderPixelGoalOverlay);
    window.addEventListener('resize', () => {
      renderPixelGoalOverlay();
      drawTrajectory();
    });
  </script>
</body>
</html>
)HTML";
}

}  // namespace go2_monitor_cpp
