import copy
from typing import Tuple, Dict, Any
from models import Observation, Action, Transaction

class FraudDetectionEnv:
    def __init__(self):
        self.current_task_level = None
        self.current_obs = None
        self.ground_truth = None
        self.step_count = 0
        self.total_reward = 0.0
        
    def reset(self, task_level: str = "easy") -> Dict:
        """Returns the initial observation based on task difficulty."""
        self.current_task_level = task_level
        self.step_count = 0
        self.total_reward = 0.0
        
        # Base normal history
        history = [
            Transaction(tx_id="t1", amount=12.50, merchant="Coffee Shop", location="New York, USA", time_since_last_tx_mins=1440),
            Transaction(tx_id="t2", amount=45.00, merchant="Grocery Store", location="New York, USA", time_since_last_tx_mins=2880),
        ]
        
        # Minimum 3 Tasks setup
        if task_level == "easy":
            # Task: Block obvious geographical anomaly and high amount
            pending = Transaction(tx_id="t3", amount=5000.00, merchant="Luxury Watches", location="Moscow, RU", time_since_last_tx_mins=5)
            self.ground_truth = "freeze_account"
            
        elif task_level == "medium":
            # Task: Detect velocity testing (small transactions rapidly)
            history.append(Transaction(tx_id="t3", amount=1.00, merchant="Online Donation", location="New York, USA", time_since_last_tx_mins=2))
            history.append(Transaction(tx_id="t4", amount=2.50, merchant="App Store", location="New York, USA", time_since_last_tx_mins=1))
            pending = Transaction(tx_id="t5", amount=900.00, merchant="Electronics Hub", location="New York, USA", time_since_last_tx_mins=1)
            self.ground_truth = "decline"
            
        elif task_level == "hard":
            # Task: Subtle account takeover (normal amount, weird merchant category, slight IP/location shift)
            history = [Transaction(tx_id="t1", amount=85.00, merchant="Tech Subscriptions", location="San Francisco, USA", time_since_last_tx_mins=1000)]
            pending = Transaction(tx_id="t2", amount=95.00, merchant="VPN Services", location="San Jose, USA", time_since_last_tx_mins=10)
            self.ground_truth = "decline"

        self.current_obs = Observation(
            pending_transaction=pending,
            user_account_history=history,
            account_age_days=365
        )
        return self.current_obs.model_dump()

    def step(self, action_dict: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Processes action, returns (observation, reward, done, info)."""
        try:
            action = Action(**action_dict)
        except Exception as e:
            # Penalize invalid formatting
            return self.current_obs.model_dump(), -1.0, True, {"error": str(e)}

        self.step_count += 1
        reward = 0.0
        
        # Meaningful Reward Function
        if action.decision == self.ground_truth:
            reward = 1.0  # Full reward for exact correct action
        elif action.decision in ["decline", "freeze_account"] and self.ground_truth in ["decline", "freeze_account"]:
            reward = 0.5  # Partial reward for recognizing fraud, but picking the wrong severity
        else:
            reward = -1.0 # Penalty for false positive (blocking good user) or false negative (allowing fraud)

        self.total_reward += reward
        done = True # Single-step trajectory for this specific pending transaction
        
        info = {
            "reasoning_provided": action.reasoning,
            "ground_truth": self.ground_truth
        }
        
        return self.current_obs.model_dump(), reward, done, info

    def state(self) -> Dict[str, Any]:
        """Returns internal environment state."""
        return {
            "task_level": self.current_task_level,
            "steps_taken": self.step_count,
            "current_reward": self.total_reward
        }

    def grade(self) -> float:
        """Deterministic grader returning 0.0 to 1.0"""
        # Normalize reward to 0.0 - 1.0 scale
        if self.total_reward >= 1.0: return 1.0
        if self.total_reward > 0.0: return 0.5
        return 0.0