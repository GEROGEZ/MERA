

from dataclasses import dataclass

@dataclass
class Case:
    ticket_id:str = None
    description: str = None
    original_severity: str = None
    prompt : str = None
    completion : str = None
    predict_severity : str = "一般"
    expert : str = None

