"""VLM API client for video and image queries."""

import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class VLMAPIClient:
    """Client for vLLM-compatible API endpoints.

    This class provides a unified interface for sending prompts with
    video, images, or text to VLM API endpoints.

    Attributes:
        endpoint: URL of the VLM API endpoint
        model: Model name to use (e.g., 'qwen2.5-vl-7b')
        temperature: Sampling temperature (0 = deterministic)
        max_tokens: Maximum tokens to generate
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: int = 0,
        max_tokens: int = 1024,
        seed: int = 42
    ):
        """Initialize VLM API client.

        Args:
            endpoint: VLM API endpoint URL
            model: Model identifier
            temperature: Sampling temperature (default: 0)
            max_tokens: Max tokens to generate (default: 1024)
            seed: Random seed (default: 42)
        """
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.headers = {"Content-Type": "application/json"}

    def send_prompt_with_video(
        self,
        prompt: str,
        video_base64: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Send a prompt with video to the VLM API.

        Args:
            prompt: User prompt/question
            video_base64: Video encoded as base64 string
            system_prompt: Optional system prompt (default: "You are a helpful assistant.")

        Returns:
            Model response text

        Raises:
            requests.RequestException: If API call fails
            ValueError: If response format is unexpected
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }

        return self._post(payload)

    def send_prompt_with_image(
        self,
        prompt: str,
        image_base64: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Send a prompt with image to the VLM API.

        Args:
            prompt: User prompt/question
            image_base64: Image encoded as base64 string
            system_prompt: Optional system prompt

        Returns:
            Model response text

        Raises:
            requests.RequestException: If API call fails
            ValueError: If response format is unexpected
        """
        # Clean base64 string and ensure proper padding
        clean_b64 = image_base64.replace("\n", "").strip()
        if len(clean_b64) % 4 != 0:
            clean_b64 += "=" * (4 - (len(clean_b64) % 4))

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"},
                        },
                    ],
                },
            ],
        }

        return self._post(payload)

    def send_text_query(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Send a text-only query to the VLM API.

        Args:
            prompt: User prompt/question
            context: Additional context text (e.g., summary report)
            system_prompt: Optional system prompt

        Returns:
            Model response text

        Raises:
            requests.RequestException: If API call fails
            ValueError: If response format is unexpected
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"{prompt}\n\n{context}"}],
                },
            ],
        }

        return self._post(payload)

    def _post(self, payload: Dict[str, Any]) -> str:
        """Execute API call with error handling.

        Args:
            payload: Request payload

        Returns:
            Model response text

        Raises:
            requests.RequestException: If API call fails
            ValueError: If response format is unexpected
        """
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()

            result = response.json()
            return str(result["choices"][0]["message"]["content"])

        except requests.RequestException as e:
            logger.error(f"VLM API request failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {e}")
            raise ValueError(f"Invalid API response format: {e}")
