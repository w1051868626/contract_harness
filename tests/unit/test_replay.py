from datetime import datetime, timezone

from harness.core.types import AgentSession, AgentStep, ContractDocument, ToolCall
from harness.replay.recorder import SessionRecorder
from harness.replay.storage import ReplayStorage


class TestReplay:
    def test_recorder_serialization(self, tmp_path):
        doc = ContractDocument(id="test", title="测试合同", content="内容")
        session = AgentSession(
            session_id="abc123",
            document=doc,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        step = AgentStep(step_index=1, agent_message="测试步骤")
        step.tool_calls.append(
            ToolCall(
                tool_name="test_tool",
                input={"key": "value"},
                output={"result": "ok"},
            )
        )
        session.steps.append(step)

        recorder = SessionRecorder()
        data = recorder._serialize(session)
        assert data["session_id"] == "abc123"
        assert data["document"]["title"] == "测试合同"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["tool_calls"][0]["tool_name"] == "test_tool"

    def test_recorder_save_load(self, tmp_path):
        doc = ContractDocument(id="test", title="测试", content="内容")
        session = AgentSession(
            session_id="save_test",
            document=doc,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        recorder = SessionRecorder()
        recorder.record(session, output_dir=tmp_path)

        storage = ReplayStorage(storage_dir=tmp_path)
        loaded = storage.load("save_test")
        assert loaded is not None
        assert loaded["session_id"] == "save_test"
