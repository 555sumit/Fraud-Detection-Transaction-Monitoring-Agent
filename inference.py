import os
import json
from openai import OpenAI
from env import FraudDetectionEnv
from models import Action

# Mandatory Environment Variables
api_base_url = os.environ.get("API_BASE_URL")
hf_token = os.environ.get("HF_TOKEN")
model_name = os.environ.get("MODEL_NAME")

# Configure Client to use specified endpoints
client = OpenAI(
    api_key=hf_token,
    base_url=api_base_url
)

def run_inference():
    env = FraudDetectionEnv()
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        # STRICT LOG FORMAT REQUIRED
        print(f"[START] task={task}")
        
        obs = env.reset(task_level=task)
        done = False
        
        system_prompt = (
            "You are a Bank Fraud Detection AI. Analyze the pending transaction against the user's history. "
            "Output strictly valid JSON matching this schema: \n"
            f"{Action.model_json_schema()}"
        )
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Observation: {json.dumps(obs)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            action_dict = json.loads(response.choices[0].message.content)
            next_obs, reward, done, info = env.step(action_dict)
            
            # STRICT LOG FORMAT REQUIRED
            print(f"[STEP] action={json.dumps(action_dict)} reward={reward}")
            
        except Exception as e:
            print(f"[STEP] action=ERROR reward=-1.0 error={str(e)}")
            
        score = env.grade()
        
        # STRICT LOG FORMAT REQUIRED
        print(f"[END] task={task} score={score}")

if __name__ == "__main__":
    run_inference()