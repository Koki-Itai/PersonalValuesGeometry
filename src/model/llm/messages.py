from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role", mode="after")
    def role_must_be_valid(cls, role: str) -> str:
        if role not in {"assistant", "system", "user"}:
            raise ValueError("role must be one of: assistant, system or user")
        return role


Messages = list[Message]
