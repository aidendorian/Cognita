from observability.langfuse import get_client

def trace_llm_call(name: str, prompt: str, response: str, model: str, metadata: dict | None = None) -> None:
    try:
        client = get_client()
        with client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=prompt,metadata=metadata or {}
        ) as generation:
            generation.update(output=response)
            
    except Exception as e:
        print(f"[langfuse] failed to trace {name}: {e}")