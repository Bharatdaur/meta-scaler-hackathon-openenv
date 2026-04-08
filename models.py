from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Instance(BaseModel):
    id: str
    type: str # 't3.micro', 'm5.large', etc.
    cpu_utilization_percent: float # 0.0 to 100.0
    status: str # 'running', 'stopped', 'terminated'
    hourly_cost: float

class Volume(BaseModel):
    id: str
    size_gb: int
    status: str # 'in-use', 'available', 'deleted'
    attached_to: Optional[str]
    monthly_cost: float

class Observation(BaseModel):
    current_hourly_total_cost: float
    instances: List[Instance]
    volumes: List[Volume]
    system_performance_score: float # 0.0 to 1.0 (drops if instance is resized too small)
    step_count: int

class Action(BaseModel):
    # Action types available to the agent
    action_type: Literal['terminate_instance', 'stop_instance', 'resize_instance', 'delete_volume', 'no_op']
    target_id: Optional[str] = Field(None, description="The ID of the instance or volume")
    new_type: Optional[str] = Field(None, description="The new instance type, required only for resize_instance (e.g. 't3.micro')")

class Reward(BaseModel):
    value: float # 0.0 to 1.0 generally
    message: str
