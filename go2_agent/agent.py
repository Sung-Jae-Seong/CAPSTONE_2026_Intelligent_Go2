#!/usr/bin/env python3.9
#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import asyncio
import os
import signal
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dotenv
import pyinputplus as pyip

LOCAL_ROSA_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rosa", "src"
)
if os.path.isdir(LOCAL_ROSA_SRC) and LOCAL_ROSA_SRC not in sys.path:
    sys.path.insert(0, LOCAL_ROSA_SRC)

from langchain.agents import tool
# from langchain_ollama import ChatOllama
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rosa import ROSA

from help import get_help
from llm import get_llm
from prompts import get_prompts
from tools import sports_api, InternVLN

instruction = ""


def update_instruction_text(new_instruction: str):
    global instruction

    if new_instruction and new_instruction.strip():
        instruction = new_instruction.strip()
        return f"Successfully updated to: {instruction}\n", 200
    return "Error: Empty instruction\n", 400


def build_instruction_query(query: str) -> str:
    query = query.strip()
    if not instruction:
        return query
    return f"<ROSA_INSTRUCTIONS>\n{instruction}\n</ROSA_INSTRUCTIONS>\n\n{query}"

class GracefulInterruptHandler:
    """Context manager to handle interrupts gracefully."""
    
    def __init__(self, verbose: bool = True):
        self.interrupted = False
        self.original_handler = None
        self.verbose = verbose
    
    def __enter__(self):
        self.interrupted = False
        self.original_handler = signal.signal(signal.SIGINT, self._handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)
        return False
    
    def _handler(self, signum, frame):
        self.interrupted = True
        if self.verbose:
            print("\n[Interrupt received - stopping current operation...]")
        # Raise KeyboardInterrupt to break out of loops
        raise KeyboardInterrupt


# Typical method for defining tools in ROSA
@tool
def ros2_helper_tool():
    """A generic ROS 2 helper tool."""
    return "This is a generic ROS 2 helper tool."


class ROS2Agent(ROSA):

    def __init__(self, streaming: bool = False, verbose: bool = True):
        self.__blacklist = ["master", "docker"]
        self.__prompts = get_prompts()
        self.__llm = get_llm(streaming=streaming)
        self._request_lock = threading.Lock()

        # self.__llm = ChatOllama(
        #     base_url="host.docker.internal:11434",
        #     model="llama3.1",
        #     temperature=0,
        #     num_ctx=8192,
        # )

        self.__streaming = streaming

        super().__init__(
            ros_version=2,
            llm=self.__llm,
            tools=[ros2_helper_tool],
            tool_packages=[sports_api, InternVLN],
            blacklist=self.__blacklist,
            prompts=self.__prompts,
            verbose=verbose,
            accumulate_chat_history=True,
            streaming=streaming,
        )

        self.examples = [
            "Give me a ROS 2 tutorial for beginners.",
            "Show me a minimal rclpy publisher and subscriber example.",
            "Give me a list of nodes, topics, services, parameters, actions, and log files.",
            "Show me how to declare and read parameters in rclpy.",
            "Show me a minimal ROS 2 service and client example.",
            "Show me a minimal ROS 2 launch file example.",
        ]

        self.command_handler = {
            "help": lambda: self.submit(get_help(self.examples)),
            "examples": lambda: self.submit(self.choose_example()),
            "clear": lambda: self.clear(),
        }

    @property
    def greeting(self):
        greeting = Text(
            "\nHi! I'm the ROSA-ROS2 agent. How can I help you today?\n"
        )
        greeting.stylize("frame bold blue")
        greeting.append(
            f"Try {', '.join(self.command_handler.keys())} or exit.",
            style="italic",
        )
        return greeting

    def choose_example(self):
        """Get user selection from the list of examples."""
        return pyip.inputMenu(
            self.examples,
            prompt="\nEnter your choice and press enter: \n",
            numbered=True,
            blank=False,
            timeout=60,
            default="1",
        )

    async def clear(self):
        """Clear the chat history."""
        self.clear_chat()
        self.last_events = []
        self.command_handler.pop("info", None)
        os.system("clear")

    def get_input(self, prompt: str):
        """Get user input from the console."""
        return pyip.inputStr(prompt, default="help")

    async def run(self):
        """
        Run the TurtleAgent's main interaction loop.

        This method initializes the console interface and enters a continuous loop to handle user input.
        It processes various commands including 'help', 'examples', 'clear', and 'exit', as well as
        custom user queries. The method uses asynchronous operations to stream responses and maintain
        a responsive interface.

        The loop continues until the user inputs 'exit'.

        Returns:
            None

        Raises:
            Any exceptions that might occur during the execution of user commands or streaming responses.
        """
        await self.clear()
        console = Console()

        while True:
            try:
                console.print(self.greeting)
                input = self.get_input("> ")

                # Handle special commands
                if input == "exit":
                    break
                elif input in self.command_handler:
                    await self.command_handler[input]()
                else:
                    await self.submit(input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation interrupted. Type 'exit' to quit or continue with a new query.[/yellow]")
                # Clear any partial state
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

    async def submit(self, query: str):
        if self.__streaming:
            await self.stream_response(query)
        else:
            self.print_response(query)

    def ask(self, query: str):
        with self._request_lock:
            return self.invoke(build_instruction_query(query))

    def print_response(self, query: str):
        """
        Submit the query to the agent and print the response to the console.

        Args:
            query (str): The input query to process.

        Returns:
            None
        """
        console = Console()
        content_panel = None

        try:
            with GracefulInterruptHandler():
                with self._request_lock:
                    response = self.invoke(build_instruction_query(query))
                with Live(
                    console=console, auto_refresh=True, vertical_overflow="visible"
                ) as live:
                    content_panel = Panel(
                        Markdown(response), title="Final Response", border_style="green"
                    )
                    live.update(content_panel, refresh=True)
        except KeyboardInterrupt:
            console.print("\n[yellow]Response interrupted.[/yellow]")
            raise

    async def stream_response(self, query: str):
        """
        Stream the agent's response with rich formatting.

        This method processes the agent's response in real-time, updating the console
        with formatted output for tokens and keeping track of events.

        Args:
            query (str): The input query to process.

        Returns:
            None

        Raises:
            Any exceptions raised during the streaming process.
        """
        console = Console()
        content = ""
        self.last_events = []

        panel = Panel("", title="Streaming Response", border_style="green")

        try:
            with GracefulInterruptHandler() as handler:
                with self._request_lock:
                    with Live(panel, console=console, auto_refresh=False) as live:
                        async for event in self.astream(build_instruction_query(query)):
                            if handler.interrupted:
                                break

                            event["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                                :-3
                            ]
                            if event["type"] == "token":
                                content += event["content"]
                                panel.renderable = Markdown(content)
                                live.refresh()
                            elif event["type"] in ["tool_start", "tool_end", "error"]:
                                self.last_events.append(event)
                            elif event["type"] == "final":
                                content = event["content"]
                                if self.last_events:
                                    panel.renderable = Markdown(
                                        content
                                        + "\n\nType 'info' for details on how I got my answer."
                                    )
                                else:
                                    panel.renderable = Markdown(content)
                                panel.title = "Final Response"
                                live.refresh()

                if self.last_events:
                    self.command_handler["info"] = self.show_event_details
                else:
                    self.command_handler.pop("info", None)
        except KeyboardInterrupt:
            console.print("\n[yellow]Response interrupted.[/yellow]")
            raise

    async def show_event_details(self):
        """
        Display detailed information about the events that occurred during the last query.
        """
        console = Console()

        if not self.last_events:
            console.print("[yellow]No events to display.[/yellow]")
            return
        else:
            console.print(Markdown("# Tool Usage and Events"))

        for event in self.last_events:
            timestamp = event["timestamp"]
            if event["type"] == "tool_start":
                console.print(
                    Panel(
                        Group(
                            Text(f"Input: {event.get('input', 'None')}"),
                            Text(f"Timestamp: {timestamp}", style="dim"),
                        ),
                        title=f"Tool Started: {event['name']}",
                        border_style="blue",
                    )
                )
            elif event["type"] == "tool_end":
                console.print(
                    Panel(
                        Group(
                            Text(f"Output: {event.get('output', 'N/A')}"),
                            Text(f"Timestamp: {timestamp}", style="dim"),
                        ),
                        title=f"Tool Completed: {event['name']}",
                        border_style="green",
                    )
                )
            elif event["type"] == "error":
                console.print(
                    Panel(
                        Group(
                            Text(f"Error: {event['content']}", style="bold red"),
                            Text(f"Timestamp: {timestamp}", style="dim"),
                        ),
                        border_style="red",
                    )
                )
            console.print()

        console.print("[bold]End of events[/bold]\n")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_streaming_setting(argv=None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    env_value = os.getenv("ROSA_STREAMING")
    streaming = _parse_bool(env_value) if env_value else False

    for index, arg in enumerate(args):
        if arg == "--no-streaming":
            return False
        if arg == "--streaming":
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                return _parse_bool(args[index + 1])
            return True
        if arg.startswith("--streaming="):
            return _parse_bool(arg.split("=", 1)[1])

    return streaming


class AgentHTTPRequestHandler(BaseHTTPRequestHandler):
    agent = None

    def do_POST(self):
        body = self._read_body()

        if self.path == "/update_instruction":
            response, status = update_instruction_text(body)
        elif self.path == "/query":
            if not body:
                response, status = "Error: Empty query\n", 400
            else:
                response = self.agent.ask(body)
                if not response.endswith("\n"):
                    response += "\n"
                status = 200
        else:
            response, status = "Error: Not found\n", 404

        self._write_response(response, status)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length).decode("utf-8").strip()

    def _write_response(self, body: str, status: int):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def create_http_server(agent):
    host = os.getenv("ROSA_HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("ROSA_HTTP_PORT", "5000"))
    AgentHTTPRequestHandler.agent = agent
    server = ThreadingHTTPServer((host, port), AgentHTTPRequestHandler)
    return server, host, server.server_address[1]


def start_http_server(agent):
    server, host, port = create_http_server(agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[HTTP] Listening on http://{host}:{port}")
    return server


def main(streaming: bool = False):
    dotenv.load_dotenv(dotenv.find_dotenv())

    ros2_agent = ROS2Agent(verbose=False, streaming=streaming)
    http_server = None

    try:
        if sys.stdin.isatty():
            http_server = start_http_server(ros2_agent)
            asyncio.run(ros2_agent.run())
        else:
            http_server, host, port = create_http_server(ros2_agent)
            print(f"[HTTP] Listening on http://{host}:{port}")
            http_server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Shutdown complete]")
    except Exception as e:
        print(f"\n[Error: {e}]")
        sys.exit(1)
    finally:
        if http_server:
            http_server.shutdown()
            http_server.server_close()


if __name__ == "__main__":
    try:
        main(streaming=get_streaming_setting())
    except KeyboardInterrupt:
        print("\n[Shutdown complete]")
