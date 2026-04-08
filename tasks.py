from typing import Dict, Any

# Instance Type definitions and their hourly costs & CPU capacities
# Arbitrary units for CPU capacity to simulate right-sizing.
INSTANCE_METADATA = {
    't3.micro': {'hourly_cost': 0.0104, 'capacity': 10},
    't3.small': {'hourly_cost': 0.0208, 'capacity': 20},
    'm5.large': {'hourly_cost': 0.096,  'capacity': 100},
    'm5.xlarge': {'hourly_cost': 0.192, 'capacity': 200},
    'c5.2xlarge': {'hourly_cost': 0.34, 'capacity': 400}
}
# Volume cost is roughly $0.10 per GB per month -> per hour = 0.10 / 730
VOLUME_COST_PER_GB_MONTH = 0.10

def _get_hourly_volume_cost(size_gb: int) -> float:
    return (size_gb * VOLUME_COST_PER_GB_MONTH) / 730.0

def load_task(task_id: str) -> Dict[str, Any]:
    if task_id == "easy":
        # Task 1: Zombie Volumes
        # Description: Delete the two available volumes, keep everything else running.
        return {
            "instances": [
                {"id": "i-098abc", "type": "t3.micro", "cpu": 45.0, "status": "running"},
                {"id": "i-283xyz", "type": "m5.large", "cpu": 60.0, "status": "running"}
            ],
            "volumes": [
                {"id": "vol-111", "size": 50, "status": "in-use", "attached_to": "i-098abc"},
                {"id": "vol-222", "size": 100, "status": "in-use", "attached_to": "i-283xyz"},
                {"id": "vol-333", "size": 500, "status": "available", "attached_to": None},
                {"id": "vol-444", "size": 1000, "status": "available", "attached_to": None}
            ],
            "max_steps": 5,
            "required_workload": 50 # sum of required CPU capacity units
        }
    elif task_id == "medium":
        # Task 2: Idle Instances
        # Description: Stop/terminate instances with 0 CPU and delete available volumes.
        return {
            "instances": [
                {"id": "i-100", "type": "m5.large", "cpu": 60.0, "status": "running"},
                {"id": "i-200", "type": "c5.2xlarge", "cpu": 0.0, "status": "running"}, # IDLE!
                {"id": "i-300", "type": "t3.small", "cpu": 0.0, "status": "running"}   # IDLE!
            ],
            "volumes": [
                {"id": "vol-1", "size": 100, "status": "in-use", "attached_to": "i-100"},
                {"id": "vol-2", "size": 500, "status": "in-use", "attached_to": "i-200"},
                {"id": "vol-3", "size": 20, "status": "in-use", "attached_to": "i-300"},
                {"id": "vol-4", "size": 200, "status": "available", "attached_to": None}
            ],
            "max_steps": 10,
            "required_workload": 60
        }
    elif task_id == "hard":
        # Task 3: Right-Sizing and Cleanup
        # Description: Complex mix of zombie volumes, idle instances, and overprovisioned instances.
        return {
            "instances": [
                {"id": "i-opt1", "type": "m5.large", "cpu": 80.0, "status": "running"}, # fine
                {"id": "i-idle1", "type": "m5.xlarge", "cpu": 0.0, "status": "running"}, # stop it
                {"id": "i-over1", "type": "c5.2xlarge", "cpu": 2.0, "status": "running"}, # downsize to t3.small
                {"id": "i-over2", "type": "m5.large", "cpu": 5.0, "status": "running"}  # downsize to t3.micro
            ],
            "volumes": [
                {"id": "vol-a", "size": 100, "status": "in-use", "attached_to": "i-opt1"},
                {"id": "vol-b", "size": 500, "status": "in-use", "attached_to": "i-idle1"},
                {"id": "vol-c", "size": 200, "status": "in-use", "attached_to": "i-over1"},
                {"id": "vol-d", "size": 100, "status": "in-use", "attached_to": "i-over2"},
                {"id": "vol-un1", "size": 1000, "status": "available", "attached_to": None},
                {"id": "vol-un2", "size": 2000, "status": "available", "attached_to": None}
            ],
            "max_steps": 15,
            "required_workload": 100 # need at least 100 capacity running!
        }
    else:
        raise ValueError(f"Unknown task: {task_id}")
