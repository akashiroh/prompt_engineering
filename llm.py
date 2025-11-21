class LLMAgent():
    """LLM Agent Class that will store an LLM and a precondition"""
    def __init__(
        self,
        llm,
        precondition: str,
        with_history: bool=False,
    ):
        self.llm = llm
        self.precondition = precondition
        self.with_history = with_history
        if with_history:
            self.history = []


    def chat(self, usr_msg, max_tokens: int=200):
        """query the llm with the user's message."""

        if self.with_history:
            prompt = [{"role": "system", "content": self.precondition,}]
            prompt |= self.history
            prompt |= [{"role": "user", "content": usr_msg,}]
        else:
            prompt = [
                {"role": "system", "content": self.precondition,},
                {"role": "user", "content": usr_msg,}
            ]

        result = self.llm.create_chat_completion(
            messages=prompt,
            max_tokens=max_tokens,
        )

        answer = result["choices"][0]["message"]
        if self.with_history:
            history.append(answer)

        return answer["content"]
