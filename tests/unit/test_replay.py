"""回放系统（录制器与存储）单元测试。"""

from datetime import datetime, timezone

from harness.core.types import AgentSession, AgentStep, ContractDocument, ToolCall
from harness.replay.player import SessionPlayer
from harness.replay.recorder import SessionRecorder
from harness.replay.storage import ReplayStorage


class TestReplay:
    """会话录制、序列化、保存与加载测试。"""

    def test_recorder_serialization(self, tmp_path):
        """会话对象应能正确序列化为字典。"""
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

    def test_storage_delete_existing(self, tmp_path):
        """删除存在的会话应返回 True 且文件不复存在。"""
        doc = ContractDocument(id="test", title="测试", content="内容")
        session = AgentSession(
            session_id="del_test", document=doc, started_at="now"
        )
        recorder = SessionRecorder()
        recorder.record(session, output_dir=tmp_path)
        storage = ReplayStorage(storage_dir=tmp_path)
        assert storage.delete("del_test") is True
        assert storage.load("del_test") is None

    def test_storage_delete_nonexistent(self, tmp_path):
        """删除不存在的会话应返回 False。"""
        storage = ReplayStorage(storage_dir=tmp_path)
        assert storage.delete("no_such_session") is False

    def test_storage_list_sessions(self, tmp_path):
        """列出会话应包含正确的摘要信息。"""
        doc1 = ContractDocument(id="doc1", title="合同A", content="内容1")
        doc2 = ContractDocument(id="doc2", title="合同B", content="内容2")
        recorder = SessionRecorder()
        recorder.record(
            AgentSession(session_id="s1", document=doc1, started_at="2026-01-01T00:00:00"),
            output_dir=tmp_path,
        )
        recorder.record(
            AgentSession(session_id="s2", document=doc2, started_at="2026-01-02T00:00:00"),
            output_dir=tmp_path,
        )
        storage = ReplayStorage(storage_dir=tmp_path)
        sessions = storage.list_sessions()
        assert len(sessions) == 2
        titles = {s["document_title"] for s in sessions}
        assert "合同A" in titles
        assert "合同B" in titles

    def test_storage_list_empty(self, tmp_path):
        """空存储目录应返回空列表。"""
        storage = ReplayStorage(storage_dir=tmp_path)
        assert storage.list_sessions() == []

    def test_player_step_through(self, tmp_path):
        """step_through 应能遍历会话步骤。"""
        doc = ContractDocument(id="test", title="测试", content="内容")
        session = AgentSession(session_id="step_test", document=doc, started_at="now")
        session.steps.append(AgentStep(step_index=1, agent_message="第一步"))
        session.steps.append(AgentStep(step_index=2, agent_message="第二步"))
        recorder = SessionRecorder()
        recorder.record(session, output_dir=tmp_path)
        player = SessionPlayer(ReplayStorage(storage_dir=tmp_path))
        steps = list(player.step_through("step_test"))
        assert len(steps) == 2
        assert steps[0].step_index == 1
        assert steps[1].agent_message == "第二步"
