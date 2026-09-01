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
    llm_backend: str = Field(..., validation_alias="LLM_BACKEND", min_length=1)
    tavily_key: str = Field(..., validation_alias="TAVILY_API_KEY", min_length=1)
    langfuse_public_key: str = Field(..., validation_alias="LANGFUSE_PUBLIC_KEY", min_length=1)
    langfuse_secret_key: str = Field(..., validation_alias="LANGFUSE_SECRET_KEY", min_length=1)
    langfuse_base_url: str = Field(..., validation_alias="LANGFUSE_BASE_URL", min_length=1)
    neo4j_uri: str = Field("bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", validation_alias="NEO4J_USER")
    neo4j_password: str = Field(..., validation_alias="NEO4J_PASSWORD", min_length=1)
    
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
llm_backend = validate.llm_backend
tavily_key = validate.tavily_key
langfuse_public_key = validate.langfuse_public_key
langfuse_secret_key = validate.langfuse_secret_key
langfuse_base_url = validate.langfuse_base_url
neo4j_uri = validate.neo4j_uri
neo4j_user = validate.neo4j_user
neo4j_password = validate.neo4j_password