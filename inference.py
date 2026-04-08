import os
import json
import time
from openai import OpenAI
from env import FinOpsEnv
from models import Action

def format_prompt(observation) -> str:
    return f"""You are a FinOps AI agent optimizing cloud costs.
Your goal is to save as much money as possible without dropping the system_performance_score below 1.0.

State:
Hourly Cost: ${observation.current_hourly_total_cost:.4f}
Performance Score: {observation.system_performance_score}
Step count: {observation.step_count}

Instances:
{json.dumps([i.model_dump() for i in observation.instances], indent=2)}

Volumes:
{json.dumps([v.model_dump() for v in observation.volumes], indent=2)}

Available actions:
1. {{\\"action_type\\": \\"terminate_instance\\", \\"target_id\\": \\"id\\"}}
2. {{\\"action_type\\": \\"stop_instance\\", \\"target_id\\": \\"id\\"}}
3. {{\\"action_type\\": \\"resize_instance\\", \\"target_id\\": \\"id\\", \\"new_type\\": \\"type\\"}} (Valid types: t3.micro, t3.small, m5.large, m5.xlarge, c5.2xlarge)
4. {{\\"action_type\\": \\"delete_volume\\", \\"target_id\\": \\"id\\"}}
5. {{\\"action_type\\": \\"no_op\\"}}

Instructions:
- Delete any volumes with status 'available'. Do NOT delete 'in-use' volumes.
- Terminate any instances with 0.0 cpu_utilization_percent (and delete their detached volumes on the next step).
- Resize instances that have very low CPU (<10%) to a smaller type to increase utilization without dropping performance.

Output exactly ONE valid JSON action object from the list above. No markdown formatting, just raw JSON.
"""

def evaluate_baseline(model_name="gpt-3.5-turbo"):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))
    env = FinOpsEnv()
    
    tasks = ["easy", "medium", "hard"]
    results = {}
    
    for task in tasks:
        print(f"\\n--- Running Baseline for Task: {task} ---")
        obs = env.reset(task)
        done = False
        
        while not done:
            prompt = format_prompt(obs)
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful JSON-only cloud cost optimization assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                
                action_str = response.choices[0].message.content.strip()
                if action_str.startswith('```json'):
                    action_str = action_str[7:-3].strip()
                elif action_str.startswith('```'):
                    action_str = action_str[3:-3].strip()
                    
                action_dict = json.loads(action_str)
            except Exception as e:
                # Fallback to no_op if API fails or parsing fails (e.g. dummy key)
                print(f"LLM Error/Fallback: {e}")
                action_dict = {"action_type": "no_op"}
                
            action = Action(**action_dict)
            
            obs, reward, done, info = env.step(action)
            print(f"Step {obs.step_count} | Action: {action.action_type} on {action.target_id} | Reward: {reward.value} | Msg: {reward.message}")
            time.sleep(0.5) # rate limit prevention
            
        final_score = env.calculate_grader_score()
        print(f"Task '{task}' Final Score: {final_score:.2f} / 1.0")
        results[task] = final_score
        
    print(f"\\n=== Baseline Results ===")
    for k, v in results.items():
        print(f"{k.capitalize()}: {v:.2f}")

if __name__ == "__main__":
    # In a real environment with API keys, would run evaluate_baseline("gpt-4o-mini")
    # For now, just print the setup
    print("Baseline script ready.")
    evaluate_baseline("gpt-3.5-turbo")
