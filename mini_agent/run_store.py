import base64
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from mini_agent.state import AgentState
from mini_agent.schemas import VisualObservation

class RunStore:
    def __init__(self,task:str):
        project_root=Path(__file__).resolve().parent.parent
        self.root_dir=project_root/"runs"
        started_at=datetime.now().astimezone()
        timestamp=started_at.strftime("%Y%m%d_%H%M%S")
        self.run_id=f"{timestamp}_{uuid4().hex[:8]}"
        self.run_dir=self.root_dir/self.run_id
        self.run_dir.mkdir(
            parents=True,
            exist_ok=False
        )
        self._write_json(
            "task.json",
            {
                "run_id":self.run_id,
                "task":task,
                "started_at":started_at.isoformat()
            }
        )

    def save_screenshot(self,step_number:int,screenshot_base64:str|None)->None:
        if screenshot_base64 is None:
            return
        screenshot_bytes=base64.b64decode(
            screenshot_base64
        )
        screenshot_path=(self.run_dir/f"step_{step_number:03d}.png")
        screenshot_path.write_bytes(
            screenshot_bytes
        )

    def save_visual_observation(self,step_number:int,observation:VisualObservation|None)->None:
        if observation is None:
            return
        self._write_json(
            f"visual_{step_number:03d}.json",
            observation.model_dump(
                mode="json"
            )
        )

    def save_state(self,state:AgentState)->None:
        state_data=state.model_dump(
            mode="json"
        )
        self._write_json(
            "state.json",
            state_data
        )
        self._write_json(
            "history.json",
            state_data["history"]
        )

    def _write_json(self,filename:str,data:object)->None:
        file_path=self.run_dir/filename
        file_path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )


































































































