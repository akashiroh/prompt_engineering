from llama_cpp import Llama

def initialize_model():
    """return a pretrained llm from huggingface."""
    llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2-7B-Instruct-GGUF",
        filename="*q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    return llm


SYSTEM = """
Your role is [ASSISTANT].

You can only answer questions about restaraunts.

You should be helpful and respectful, if the USER asks anything that you are unsure about respond with 'I'm sorry, I don't know'.
"""

history = [
    {"role": "system", "content": SYSTEM,}
]

def chat(llm, usr_msg):
    """query the llm with the user's message. support multi turn conversations."""

    history.append({"role": "user", "content": usr_msg})

    result = llm.create_chat_completion(
        messages = history,
        max_tokens=200,
    )

    # answer = result["choices"][0]["text"].strip()
    answer = result["choices"][0]["message"]
    history.append(answer)

    return answer["content"]

def main():
    """conversation loop."""
    llm = initialize_model()

    print("Query the LLM.")
    usr_msg = None

    while usr_msg != " ":
        usr_msg = input()
        if usr_msg == "DEBUG":
            import ipdb
            ipdb.set_trace()
            continue
        response = chat(llm, usr_msg)
        print("\n[ASSISTANT]:", response)
        print()



if __name__ == "__main__":
    main()
