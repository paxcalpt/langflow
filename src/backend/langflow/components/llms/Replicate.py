from langflow import CustomComponent
from typing import Optional, Union, Callable
from langflow.field_typing import BaseLanguageModel
from langchain_community.llms.replicate import Replicate
import os


class ReplicateComponent(CustomComponent):
    display_name = "Replicate"
    description = "`Replicate` large language models."
    documentation = "https://python.langchain.com/docs/integrations/llms/replicate"

    def build_config(self):
        return {
            "model": {
                "display_name": "Model name",
                "field_type": "str",
                "advanced": False,
                "required": True,
                "default": "mistralai/mistral-7b-instruct-v0.2:f5701ad84de5715051cb99d550539719f8a7fbcf65e0e62a3d1eb3f94720764e",
            },
            "model_kwargs": {
                "display_name": "Model Keyword Arguments",
                "field_type": "dict",
                "advanced": True,
                "required": False,
            },
            "replicate_api_token": {
                "display_name": "Replicate API Token",
                "field_type": "str",
                "advanced": False,
                "required": True,
                "password": True,
            },
            "prompt_key": {
                "display_name": "Prompt Key",
                "field_type": "str",
                "advanced": True,
                "required": False,
            },
            "streaming": {
                "display_name": "Streaming",
                "field_type": "bool",
                "advanced": True,
                "required": False,
                "default": True,
            },
            "stop": {
                "display_name": "Stop Sequences",
                "field_type": "list",
                "advanced": True,
                "required": False,
            },
        }

    def build(
        self,
        model: str,
        model_kwargs: Optional[dict] = {},
        replicate_api_token: Optional[str] = None,
        prompt_key: Optional[str] = None,
        streaming: bool = False,
        stop: Optional[list] = [],
    ) -> Union[BaseLanguageModel, Callable]:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
        try:
            import replicate as replicate_python
        except ImportError:
            raise ImportError(
                "Could not import replicate python package. "
                "Please install it with `pip install replicate`."
            )
        return Replicate(
            model=model,
            model_kwargs=model_kwargs,
            replicate_api_token=replicate_api_token,
            prompt_key=prompt_key,
            streaming=streaming,
            stop=stop,
        )
