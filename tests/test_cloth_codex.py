from copy import deepcopy
import json
import sys

from cloth_agent.codex import CodexAppServer, tool_specs
from cloth_agent.config import MOCK_CONFIG, load_config
from cloth_agent.runtime import Runtime


def test_tool_bridge_handles_jsonrpc_images_and_errors(tmp_path):
    # A protocol peer checks actual requests and response shapes without a
    # model dependency. The real app-server/model smoke run is a separate CLI.
    peer = tmp_path / "peer.py"
    peer.write_text('''import json, sys
def receive():
    return json.loads(sys.stdin.readline())
def send(value):
    print(json.dumps(value), flush=True)
message = receive()
assert message["method"] == "initialize"
assert message["params"]["capabilities"]["experimentalApi"]
send({"id": message["id"], "result": {}})
assert receive()["method"] == "initialized"
message = receive()
assert message["method"] == "thread/start"
assert all(tool["type"] == "function" for tool in message["params"]["dynamicTools"])
assert "中文" in message["params"]["developerInstructions"]
send({"id": message["id"], "result": {"thread": {"id": "test-thread"}}})
message = receive()
assert message["method"] == "turn/start"
assert message["params"]["threadId"] == "test-thread"
send({"id": message["id"], "result": {"turn": {"id": "test-turn"}}})
send({"method": "item/agentMessage/delta", "params": {"delta": "准备读取 RGB-D 观测。\\n"}})
send({"id": 100, "method": "item/tool/call", "params": {
    "tool": "cloth_observe", "arguments": {}}})
response = receive()["result"]
assert response["success"]
assert response["contentItems"][0]["type"] == "inputText"
assert response["contentItems"][1]["type"] == "inputImage"
send({"id": 101, "method": "item/tool/call", "params": {
    "tool": "cloth_finish_task", "arguments": {}}})
assert not receive()["result"]["success"]  # no successful programs
send({"id": 102, "method": "item/tool/call", "params": {
    "tool": "cloth_abort_task", "arguments": {"reason": "协议联调 protocol test"}}})
assert receive()["result"]["success"]
send({"method": "item/agentMessage/delta", "params": {"delta": "任务已放弃 task aborted。\\n"}})
send({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})
for line in sys.stdin:
    pass
''')
    config = deepcopy(load_config(MOCK_CONFIG))
    config["codex"].update(executable=sys.executable, extra_args=[str(peer)], request_timeout_s=5)
    with Runtime(config, tmp_path / "run") as runtime:
        output = []
        result = CodexAppServer(runtime, progress=output.append).run("协议联调 protocol test")
        assert result["status"] == "aborted"
        assert json.loads((runtime.directory / "thread.json").read_text())["thread_id"] == "test-thread"
        record = (runtime.directory / "codex.md").read_text(encoding="utf-8")
        assert "准备读取 RGB-D 观测。" in record
        assert "任务已放弃 task aborted。" in record
        assert "协议联调 protocol test" in record
        assert "工具错误 Tool error" in record  # Failed finish_task remains visible.
        assert "cloth_finish_task" in record
        assert "cloth_abort_task" in record
        assert "准备读取 RGB-D 观测。" in "".join(output)
        events = [json.loads(line) for line in (runtime.directory / "events.jsonl").read_text().splitlines()]
        assert all("description" in event for event in events)
        assert any(event["type"] == "task_aborted" for event in events)
        assert "已放弃" in result["status_text"]


def test_geometry_results_do_not_require_an_image_envelope(tmp_path):
    with Runtime(load_config(MOCK_CONFIG), tmp_path / "run") as runtime:
        client = CodexAppServer(runtime)
        result = client._dispatch("cloth_geometry", {"operation": "offset",
                                   "values": {"point": [0, 0, 0], "offset": [1, 2, 3]}})
        assert result == [1, 2, 3]
        assert client.image_content(result) == []
    assert "cloth_submit_subtask" in {tool["name"] for tool in tool_specs()}
