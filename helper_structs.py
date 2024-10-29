from enum import Enum
from typing_extensions import TypedDict

class ConversationState(Enum):
    QUESTION = "question"
    ACTION = "action"
    END = "end"
    INFORMATION = "information"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    ACTION_REQUEST = "action_request"
    
class Discovery(TypedDict):
    state: ConversationState
    text: str