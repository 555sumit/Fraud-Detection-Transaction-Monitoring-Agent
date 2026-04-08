from pydantic import BaseModel, Field
from typing import List, Literal

class Transaction(BaseModel):
    tx_id: str
    amount: float
    merchant: str
    location: str
    time_since_last_tx_mins: int

class Observation(BaseModel):
    pending_transaction: Transaction
    user_account_history: List[Transaction]
    account_age_days: int

class Action(BaseModel):
    decision: Literal["approve", "decline", "freeze_account"] = Field(
        description="Approve normal behavior, decline suspicious transactions, freeze for severe account takeovers."
    )
    reasoning: str = Field(
        description="Brief explanation of why this decision was made."
    )