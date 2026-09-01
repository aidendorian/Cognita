from langfuse import Langfuse
from config.env import langfuse_public_key, langfuse_secret_key, langfuse_base_url

_client = Langfuse(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    base_url=langfuse_base_url,
)

def get_client() -> Langfuse:
    return _client