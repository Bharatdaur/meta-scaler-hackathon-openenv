---
title: FinOps Cloud Cost Optimizer
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
---
# ☁️ FinOps AI: OpenEnv for Cloud Cost Optimizer

## Environment Description & Motivation
Cloud computing bills are notoriously complex and inflated. FinOps (Financial Operations) teams spend countless hours trying to identify zombie resources (unattached storage), idle compute instances, and over-provisioned servers. 
This environment simulated a cloud architecture where an AI agent must analyze compute instances and storage volumes to reduce hourly costs as much as theoretically possible **without causing the system workload to drop below a required threshold**. 

It perfectly satisfies the criteria of "genuine real-world task" because right-sizing cloud instances is a ubiquitous corporate challenge.

## Action & Observation Spaces
### Action Models (Pydantic)
*   **Action Types:** `terminate_instance`, `stop_instance`, `resize_instance`, `delete_volume`, `no_op`
*   **Parameters:** `target_id` (str), `new_type` (str - only if resizing)

### Observation Models (Pydantic)
*   **Total Cost:** $ per hour (float)
*   **Performance Score:** 0.0 to 1.0. Drops to 0.5 or 0.0 if instances are terminated when they shouldn't be or resized below minimum workload capacity.
*   **Instances List:** Contains ID, Type, CPU utilization %, and Status.
*   **Volumes List:** Contains ID, Size, Status, and attached compute ID.

## Task Difficulty Progression
*   **Easy:** Find and delete `available` (unattached) storage volumes. Introduces agent to the state space.
*   **Medium:** Identify specifically instances that have 0.0% CPU (idle), stop them, and clean up their associated storage volumes.
*   **Hard:** Mix of idle instances, zombie volumes, and *overprovisioned* instances (e.g. `c5.2xlarge` running at 2% CPU). Agent must right-size the clusters using math to pick the cheapest instance that can support the current workload constraint without triggering the performance penalty constraint.

## Grading & Reward System
*   **Dense Signal:** Intermediate rewards are returned on each step based on cost-savings deltas and penalties for invalid API choices.
*   **Final Grader:** Computes the theoretical minimum optimal cost for the specific environment graph without performance drops. Score is plotted strictly between `0.0` (failed to optimize / performance tanked) and `1.0` (optimal right-sized cloud infrastructure achieved).

## Setup & Execution Instructions
1.  **Run Environment Wrapper:**
    ```bash
    pip install -r requirements.txt
    uvicorn app:app --port 7860
    ```
2.  **Evaluate Baseline Script:**
    ```bash
    export OPENAI_API_KEY="your-key"
    python baseline.py
    ```

## HF Spaces Container
Cleanly builds with:
```bash
docker build -t openenv-finops .
docker run -p 7860:7860 openenv-finops
```
