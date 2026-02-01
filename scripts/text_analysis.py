import tiktoken
from openai import OpenAI, RateLimitError

from .openai_decorator import retry_on_openai_errors


class AI:
    def __init__(
        self,
        encoding_model: str = "cl100k_base",
        openai_model: str = "gpt-4o-mini",
    ):
        """
        Initialize an AI instance.

        Parameters:
            encoding_model (str): The name of the encoding model to be used.
            openai_model (str): The name of the OpenAI model to be used.
        """
        self.tt_encoding = tiktoken.get_encoding(encoding_model)
        self.openai_model = openai_model
        self.client = OpenAI()

    def token_counter(self, passage):
        """
        Count the number of tokens in a given passage.

        Parameters:
            passage (str): The input text passage.

        Returns:
            int: The total number of tokens in the passage.
        """
        tokens = self.tt_encoding.encode(passage)
        total_tokens = len(tokens)
        return total_tokens

    @retry_on_openai_errors(max_retry=7)
    def extract_dialogue(self, transcript, history=None):
        """
        Extract dialogue involving multiple speaker from text.

        Parameters:
            transcript (str): The text containing the conversation.
            history (list): List of message history (optional).

        Returns:
            str: Extracted dialogue in the specified format.
        """
        prompt = """Perform speaker diarization on the given text to identify and extract conversations involving multiple speakers. Present the dialogue in the following structured format:
        Speaker 1:
        Speaker 2:
        Speaker 3:
        ..."""

        while True:
            try:
                if history:
                    messages = list(history)
                else:
                    messages = [
                        {"role": "system", "content": prompt},
                    ]
                user_message = {"role": "user",
                                "content": transcript.replace('\n', '')}
                messages.append(user_message)
                tokens_per_message = 4
                prompt_tokens = self.token_counter(prompt)
                transcript_tokens = self.token_counter(transcript)
                overhead_tokens = (len(messages) * tokens_per_message) + 3
                available_tokens = 8191 - (prompt_tokens + transcript_tokens + overhead_tokens)
                max_token = max(1, min(4096, available_tokens))
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=max_token,
                    temperature=1,
                    top_p=1,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
                bot_response = response.choices[0].message.content.strip()
                return bot_response

            except RateLimitError:
                messages.pop(1)
                continue
