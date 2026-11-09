import json
from enum import StrEnum
from pydantic import BaseModel,ConfigDict,Field
from mini_agent.schemas import AgentStepRecord

class AgentStatus(StrEnum):
    READY="ready"
    RUNNING="running"
    COMPLETED="completed"
    FAILED="failed"
    STOPPED="stopped"

class AgentState(BaseModel):
    model_config=ConfigDict(
        extra="forbid"
    )
    task:str=Field(min_length=1)
    status:AgentStatus=AgentStatus.READY
    current_step:int=0
    consecutive_failures:int=0
    repeated_action_count:int=0
    last_action_signature:str|None=None
    final_answer:str|None=None
    stop_reason:str|None=None
    history:list[AgentStepRecord]=Field(
        default_factory=list
    )
    def mark_running(self)->None:
        self.status=AgentStatus.RUNNING

    def add_record(self,record:AgentStepRecord)->None:
        self.current_step=record.step_number
        self.history.append(record)
        self._update_failure_count(record)
        self._update_repeated_action(record)
        if record.result.is_done:
            self.final_answer=record.result.answer
            if record.result.success:
                self.status=AgentStatus.COMPLETED
            else:
                self.status=AgentStatus.FAILED

    def _update_failure_count(self,record:AgentStepRecord)->None:
        if record.result.success:
            self.consecutive_failures=0
        else:
            self.consecutive_failures+=1

    def _update_repeated_action(self,record:AgentStepRecord)->None:
        action_data=(
            record.decision.action.model_dump(
                mode="json"
            )
        )
        signature=json.dumps(
            action_data,
            ensure_ascii=False,
            sort_keys=True
        )
        if signature==self.last_action_signature:
            self.repeated_action_count+=1
        else:
            self.repeated_action_count=1
            self.last_action_signature=signature

    def check_stop_policy(self,max_failures:int=3,max_repeated_actions:int=3)->str|None:
        if self.consecutive_failures>=max_failures:
            return (
                f"连续失败达到{max_failures}"
            )
        if(self.repeated_action_count>=max_repeated_actions):
            return (
                f"相同动作连续执行"
                f"{max_repeated_actions}次"
            )
        return None

    def mark_failed(self,reason:str)->None:
        self.status=AgentStatus.FAILED
        self.stop_reason=reason

    def mark_stopped(self,reason:str="用户主动停止")->None:
        self.status=AgentStatus.STOPPED
        self.stop_reason=reason
        






