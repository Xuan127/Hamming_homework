from __future__ import annotations
from enum import Enum
from typing_extensions import TypedDict
from typing import Optional, List
from pydantic import BaseModel

class ConversationState(Enum):
    QUESTION = "question"
    ACTION = "action"
    END = "end"
    INFORMATION = "information"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    ACTION_REQUEST = "action_request"
    TRANSFER = "transfer"
    FILLER = "filler"
    
class Discovery(BaseModel):
    state: str
    text: str

class Option:
    def __init__(self, option: str, next_step: Optional['NextStep'] = None):
        self.option = option
        self.next_step = next_step

class NextStep(BaseModel):
    action: Optional[str] = None
    end: bool
    next_step: Optional['NextStep'] = None

# Update forward references if needed
NextStep.update_forward_refs()
Discovery.update_forward_refs()
