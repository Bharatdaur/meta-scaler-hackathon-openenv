from typing import Tuple, Dict, Any
from models import Observation, Action, Reward, Instance, Volume
from tasks import load_task, INSTANCE_METADATA, _get_hourly_volume_cost

class FinOpsEnv:
    def __init__(self):
        self.current_task_id = None
        self._state = None
        self.max_steps = 0
        self.step_count = 0
        self.required_workload = 0

    def reset(self, task_id: str = "easy") -> Observation:
        self.current_task_id = task_id
        task_data = load_task(task_id)
        
        self.max_steps = task_data["max_steps"]
        self.step_count = 0
        self.required_workload = task_data["required_workload"]
        
        instances = []
        for inst in task_data["instances"]:
            instances.append(Instance(
                id=inst["id"],
                type=inst["type"],
                cpu_utilization_percent=inst["cpu"],
                status=inst["status"],
                hourly_cost=INSTANCE_METADATA[inst["type"]]["hourly_cost"]
            ))
            
        volumes = []
        for vol in task_data["volumes"]:
            volumes.append(Volume(
                id=vol["id"],
                size_gb=vol["size"],
                status=vol["status"],
                attached_to=vol["attached_to"],
                monthly_cost=vol["size"] * 0.10
            ))
            
        # Initial score is 1.0, cost is calculated
        self._state = Observation(
            current_hourly_total_cost=self._calculate_total_cost(instances, volumes),
            instances=instances,
            volumes=volumes,
            system_performance_score=self._calculate_performance(instances),
            step_count=self.step_count
        )
        # Store starting cost to compute relative savings for reward
        self.start_cost = self._state.current_hourly_total_cost
        return self._state

    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment not reset.")
        return self._state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not reset.")
        
        self.step_count += 1
        message = ""
        reward_value = 0.0 # sparse + shaped reward
        
        instances = {inst.id: inst.model_copy() for inst in self._state.instances}
        volumes = {vol.id: vol.model_copy() for vol in self._state.volumes}

        # Validate target
        valid_target = False
        if action.target_id in instances or action.target_id in volumes or action.action_type == 'no_op':
            valid_target = True

        if not valid_target:
            message = f"Invalid target_id: {action.target_id}"
            reward_value = -0.1
        else:
            if action.action_type == 'terminate_instance' or action.action_type == 'stop_instance':
                if action.target_id in instances:
                    inst = instances[action.target_id]
                    if inst.status != 'running':
                        message = f"Instance {action.target_id} is already {inst.status}."
                        reward_value = -0.05
                    else:
                        inst.status = 'terminated' if action.action_type == 'terminate_instance' else 'stopped'
                        inst.hourly_cost = 0.0
                        
                        # Detach attached volumes
                        for vol in volumes.values():
                            if vol.attached_to == action.target_id:
                                vol.status = 'available'
                                vol.attached_to = None
                                
                        message = f"Instance {action.target_id} {inst.status}."
                        # Reward if CPU was 0, penalize if we killed a needed instance? 
                        # We handle this via the performance score computation at the end of step
                else:
                    message = "Target is not an instance."
                    reward_value = -0.1

            elif action.action_type == 'resize_instance':
                if action.target_id in instances:
                    inst = instances[action.target_id]
                    if inst.status != 'running':
                        message = f"Cannot resize {inst.status} instance."
                        reward_value = -0.05
                    elif action.new_type not in INSTANCE_METADATA:
                        message = f"Unknown instance type {action.new_type}."
                        reward_value = -0.1
                    else:
                        old_type = inst.type
                        inst.type = action.new_type
                        inst.hourly_cost = INSTANCE_METADATA[action.new_type]['hourly_cost']
                        # Adjust simulated CPU usage roughly
                        old_cap = INSTANCE_METADATA[old_type]['capacity']
                        new_cap = INSTANCE_METADATA[action.new_type]['capacity']
                        # cpu% * old_cap = actual_workload -> new_cpu% = actual_workload / new_cap
                        actual_workload = (inst.cpu_utilization_percent / 100.0) * old_cap
                        new_cpu_percent = (actual_workload / new_cap) * 100.0
                        if new_cpu_percent > 100.0:
                            message = f"Resized {action.target_id} to {action.new_type}. Warning: Overloaded!"
                        else:
                            message = f"Resized {action.target_id} from {old_type} to {action.new_type}."
                        inst.cpu_utilization_percent = min(new_cpu_percent, 100.0)
                else:
                    message = "Target is not an instance."
                    reward_value = -0.1

            elif action.action_type == 'delete_volume':
                if action.target_id in volumes:
                    vol = volumes[action.target_id]
                    if vol.status == 'in-use':
                        message = f"Volume {action.target_id} is in-use. Cannot delete."
                        reward_value = -0.2 # severe penalty
                    elif vol.status == 'deleted':
                        message = f"Volume {action.target_id} is already deleted."
                        reward_value = -0.05
                    else:
                        vol.status = 'deleted'
                        vol.monthly_cost = 0.0
                        message = f"Volume {action.target_id} deleted."
                else:
                    message = "Target is not a volume."
                    reward_value = -0.1
            
            elif action.action_type == 'no_op':
                message = "No operation performed."

        # Compute new state
        inst_list = list(instances.values())
        vol_list = list(volumes.values())
        
        new_cost = self._calculate_total_cost(inst_list, vol_list)
        performance = self._calculate_performance(inst_list)
        
        self._state = Observation(
            current_hourly_total_cost=new_cost,
            instances=inst_list,
            volumes=vol_list,
            system_performance_score=performance,
            step_count=self.step_count
        )

        done = self.step_count >= self.max_steps
        
        # Grading / Reward function
        # A 0.0 to 1.0 grader score logic is provided at 'done'
        if done:
            score = self.calculate_grader_score()
            # Reward is exactly the score at terminal state, padded with intermediate shape
            reward_value = score
        else:
            # Intermediate reward roughly tracks cost savings over time if performance is ok
            if performance < 1.0:
                reward_value -= 0.1 # penalty for degraded system
            else:
                cost_savings = self.start_cost - new_cost
                if cost_savings > 0:
                    # Give small fraction for savings steps
                    pass

        reward = Reward(value=reward_value, message=message)
        info = {
            "performance": performance,
            "cost_savings": self.start_cost - new_cost
        }

        return self._state, reward, done, info

    def _calculate_total_cost(self, instances: list, volumes: list) -> float:
        cost = 0.0
        for i in instances:
            cost += i.hourly_cost
        for v in volumes:
            cost += (v.monthly_cost / 730.0) # hourly approx
        return cost

    def _calculate_performance(self, instances: list) -> float:
        total_capacity = sum(INSTANCE_METADATA[i.type]["capacity"] for i in instances if i.status == 'running')
        if total_capacity >= self.required_workload:
            # Check for overloaded individual instances
            for i in instances:
                if i.status == 'running' and i.cpu_utilization_percent >= 90.0:
                    return 0.5 # degraded
            return 1.0
        else:
            return 0.0 # severely degraded/down

    def calculate_grader_score(self) -> float:
        # 0.0 to 1.0 score
        # 1. Performance must be 1.0
        # 2. Maximum cost savings achieved.
        perf = self._state.system_performance_score
        if perf < 1.0:
             return 0.0 # Fail if performance drops
        
        # Calculate optimal cost based on task logic
        optimal_cost = 0.0
        if self.current_task_id == "easy":
            # Optimal: 2 instances running, 2 volumes in-use. Available volumes deleted.
            optimal_cost = (0.0104 + 0.096) + (50 * 0.1/730) + (100 * 0.1/730)
        elif self.current_task_id == "medium":
            # Optimal: 1 instance running, 1 volume in-use.
            optimal_cost = 0.096 + (100 * 0.1/730)
        elif self.current_task_id == "hard":
            # Optimal: 1 m5.large running (80%), 1 new t3.small (over1 was c5.2xl, now t3.small), 1 new t3.micro (over2 was m5.large, now t3.micro). Idle1 terminated.
            # Volumes: vol-a (100), vol-c (200), vol-d (100). vol-b and un1, un2 deleted.
            optimal_cost = 0.096 + 0.0208 + 0.0104 + ((100+200+100) * 0.1/730)
            
        current_cost = self._state.current_hourly_total_cost
        
        # Score is proportional to how close it is to optimal cost
        # If current_cost == optimal_cost -> 1.0
        # If current_cost == start_cost -> 0.0 (or negative, but floored at 0)
        if self.start_cost <= optimal_cost:
            return 1.0 # Edge case, already optimal

        savings_achieved = self.start_cost - current_cost
        max_possible_savings = self.start_cost - optimal_cost
        
        score = savings_achieved / max_possible_savings
        return max(0.0, min(1.0, score))
