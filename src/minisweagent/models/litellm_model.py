import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("litellm_model")
litellm.return_response_headers = True
# SET_COOKIE_ID = "set-cookie"
SET_COOKIE_ID = "set-cookie"  # Use standard HTTP header name for receiving cookies


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | None = None


class LitellmModel:
    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.response_headers = None
        self.x_client_id = str(uuid.uuid4())

        if self.config.litellm_model_registry is not None:
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

        from nemo_gym.server_utils import ServerClient
        from nemo_gym.global_config import get_global_config_dict
        self.server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=get_global_config_dict(),
        )

        self.model_server_cookies = None

    def _add_tokens_ids_to_messages(self, messages: list[dict[str, str]], responses: list[dict[str, str]]):
        processed_messages = []
        responses_idx = 0
        for message in messages:
            if message["role"] in ["system", "user"]:
                processed_messages.append(message)
            elif message["role"] == "assistant":
                assistant_message = message.copy()
                response = responses[responses_idx]
                if response.get("provider_specific_fields", {}):
                    provider_specific_fields = response["provider_specific_fields"]
                    for key in ["prompt_token_ids", "generation_token_ids", "generation_log_probs"]:
                        if key in provider_specific_fields:
                            assistant_message[key] = provider_specific_fields[key]
                responses_idx += 1
                processed_messages.append(assistant_message)

        return processed_messages

    async def _query(self, messages: list[dict[str, str]], responses: list[dict[str, str]], **kwargs):
        from pydantic import ValidationError

        from nemo_gym.server_utils import raise_for_status
        from nemo_gym.openai_utils import NeMoGymResponse

        body = dict(
            model=self.config.model_name,
            messages=self._add_tokens_ids_to_messages(messages, responses),
            **(self.config.model_kwargs | kwargs),
        )
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/chat/completions",
            json=body,
            cookies=self.model_server_cookies,
        )
        # We raise for status here since we expect model calls to always work.
        await raise_for_status(model_response)
        model_response_json = await model_response.json()
        self.model_server_cookies = self.model_server_cookies or model_response.cookies
        try:
            model_response = NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from model server: {json.dumps(model_response_json)}"
            ) from e

        return model_response

    async def query(self, messages: list[dict[str, str]], responses: list[dict[str, str]], **kwargs) -> dict:
        response = await self._query(messages, responses, **kwargs)
        if hasattr(response.choices[0].message, "provider_specific_fields"):
            provider_specific_fields = response.choices[0].message.provider_specific_fields
        else:
            provider_specific_fields = {}
        try:
            cost = litellm.cost_calculator.completion_cost(response)
            self.n_calls += 1
            self.cost += cost
            GLOBAL_MODEL_STATS.add(cost)
        except Exception:
            self.n_calls += 1
            self.cost += 0
            GLOBAL_MODEL_STATS.add(0)

        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "response_obj": response.model_dump() | {"provider_specific_fields": provider_specific_fields},
        }
