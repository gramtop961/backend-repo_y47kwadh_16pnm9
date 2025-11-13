"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any

class Analysis(BaseModel):
    """
    Stores each analysis run (fake news or plagiarism)
    Collection name: "analysis"
    """
    type: Literal["fake_news", "plagiarism"] = Field(..., description="Type of analysis performed")
    title: Optional[str] = Field(None, description="Optional title or headline")
    source_url: Optional[str] = Field(None, description="Optional source URL")
    text: str = Field(..., description="Input text that was analyzed")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    verdict: str = Field(..., description="High-level verdict or label")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional structured details for UI")

# Example of other collections you might add later
class User(BaseModel):
    name: str
    email: str
    organization: Optional[str] = None
