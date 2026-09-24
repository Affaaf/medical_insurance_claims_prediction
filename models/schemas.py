from pydantic import BaseModel, Field
from typing import Optional


class RawInput(BaseModel):
    text: str = Field(
        ...,
        description="Raw input text for claim prediction")


class Claim(BaseModel):
    InsuID: Optional[str] = None
    InsuPlanID: Optional[str] = None
    Diag1: Optional[str] = None
    Diag2: Optional[str] = None
    Diag3: Optional[str] = None
    Diag4: Optional[str] = None
    Diag5: Optional[str] = None
    Diag6: Optional[str] = None
    Diag7: Optional[str] = None
    Diag8: Optional[str] = None
    Diag9: Optional[str] = None
    Diag10: Optional[str] = None
    Diag11: Optional[str] = None
    Diag12: Optional[str] = None
    
    Proc1: Optional[str] = None
    Modifier1: Optional[str] = None
    Paid1: int = Field(0, ge=0, le=1)
    Proc1_Adj: Optional[str] = None
    
    Proc2: Optional[str] = None
    Modifier2: Optional[str] = None
    Paid2: int = Field(0, ge=0, le=1)
    Proc2_Adj: Optional[str] = None
    
    Proc3: Optional[str] = None
    Modifier3: Optional[str] = None
    Paid3: int = Field(0, ge=0, le=1)
    Proc3_Adj: Optional[str] = None
    
    Proc4: Optional[str] = None
    Modifier4: Optional[str] = None
    Paid4: int = Field(0, ge=0, le=1)
    Proc4_Adj: Optional[str] = None
    
    Proc5: Optional[str] = None
    Modifier5: Optional[str] = None
    Paid5: int = Field(0, ge=0, le=1)
    Proc5_Adj: Optional[str] = None
    
    Proc6: Optional[str] = None
    Modifier6: Optional[str] = None
    Paid6: int = Field(0, ge=0, le=1)
    Proc6_Adj: Optional[str] = None
