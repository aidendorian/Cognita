from pydantic import Field, ValidationError
from pydantic_settings import SettingsConfigDict, BaseSettings

class ValidateENV(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding="utf-8",
                                      env_ignore_empty=True)

    api_key: str = Field(..., validation_alias="GEMINI_API_KEY", min_length=1)
    llm_model: str = Field(..., validation_alias="LLM_MODEL", min_length=1)
    db_username: str = Field(..., validation_alias="DATABASE_USERNAME", min_length=1)
    db_password: str = Field(..., validation_alias="DATABASE_PASSWORD", min_length=1)
    db_name: str = Field(..., validation_alias="DATABASE_NAME", min_length=1)

    @property
    def db_url(self) -> str:
        return f"postgresql://{self.db_username}:{self.db_password}@localhost:5432/{self.db_name}"
    
try:
    validate = ValidateENV()  # type: ignore
except ValidationError as e:
    print(".env not properly configured\n")
    for error in e.errors():
        loc = error["loc"][0]
        error_type = error["type"]
        if error_type == "missing":
            print(f"Missing: '{loc}' must be set.")
        elif error_type == "string_too_short":
            print(f"Empty: '{loc}' cannot be empty.")
        else:
            print(f"  {loc}: {error['msg']}")
    raise SystemExit(1)

api_key = validate.api_key
llm_model = validate.llm_model
db_url = validate.db_url