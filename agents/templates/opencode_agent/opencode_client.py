import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger()
logging.getLogger("httpx").setLevel(logging.CRITICAL)


class OpenCodeClientError(Exception):
    pass


class OpenCodeClient:
    
    def __init__(
        self,
        base_url: str = "http://localhost:4096",
        timeout: float = 300.0,
        max_retries: int = 3
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            follow_redirects=True
        )
        logger.info(f"OpenCodeClient initialized: {base_url}")
    
    def _request(
        self,
        method: str,
        path: str,
        log_request: bool = True,
        **kwargs: Any
    ) -> httpx.Response:
        for attempt in range(self.max_retries):
            try:
                response = self.client.request(method, path, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                logger.error(f"HTTP {status} error on {method} {path}: {e.response.text[:200]}")
                if status == 404:
                    logger.warning("Endpoint not found - may indicate OpenCode server issue or version mismatch")
                elif status == 500:
                    logger.error("OpenCode server internal error - check server logs")
                elif status == 401 or status == 403:
                    logger.error("Authentication failure - check OpenRouter API key configuration")
                elif status == 429:
                    logger.error("Rate limit exceeded - consider adding delays between requests")
                if attempt == self.max_retries - 1:
                    raise OpenCodeClientError(f"Failed after {self.max_retries} attempts: {e}")
                logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
            except httpx.TimeoutException as e:
                logger.error(f"Timeout on {method} {path}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("Repeated timeouts may indicate server overload or network issues")
                    raise OpenCodeClientError(f"Failed after {self.max_retries} attempts: {e}")
                logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
            except httpx.RequestError as e:
                logger.error(f"Request error on {method} {path}: {e}")
                if "Connection refused" in str(e):
                    logger.error("Connection refused - OpenCode server may not be running on specified port")
                elif "Connection reset" in str(e):
                    logger.warning("Connection reset - server may be restarting or overloaded")
                if attempt == self.max_retries - 1:
                    raise OpenCodeClientError(f"Failed after {self.max_retries} attempts: {e}")
                logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
        
        raise OpenCodeClientError(f"Request failed after {self.max_retries} attempts")
    
    def health_check(self) -> dict[str, Any]:
        response = self._request("GET", "/global/health")
        return response.json()
    
    def create_session(self, title: str, parent_id: Optional[str] = None) -> dict[str, Any]:
        body: dict[str, Any] = {"title": title}
        if parent_id:
            body["parentID"] = parent_id
        
        logger.info(f"Creating session: {title}")
        response = self._request("POST", "/session", json=body)
        session = response.json()
        logger.info(f"Session created: {session.get('id')}")
        return session
    
    def send_message(
        self,
        session_id: str,
        prompt: str,
        model: Optional[dict[str, str]] = None,
        no_reply: bool = False,
        tools: Optional[dict[str, bool]] = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": prompt}]
        }
        
        if model:
            body["model"] = model
        
        if no_reply:
            body["noReply"] = True
        
        if tools:
            body["tools"] = tools
        
        logger.debug(f"Sending message to session {session_id} (prompt length: {len(prompt)})")
        
        try:
            response = self._request("POST", f"/session/{session_id}/message", json=body)
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {e}")
            raise
        
        if not response.content:
            logger.error(f"Empty response from /session/{session_id}/message - possible OpenRouter auth issue")
            logger.error(f"Response status: {response.status_code}, headers: {dict(response.headers)}")
            raise ValueError("OpenCode returned empty response body")
        
        try:
            result = response.json()
            if not result:
                logger.warning("send_message returned empty JSON object")
            return result
        except Exception as e:
            logger.error(f"Failed to parse response JSON: {e}")
            logger.error(f"Response content (first 500 chars): {response.text[:500]}")
            raise
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        log_request: bool = True
    ) -> list[dict[str, Any]]:
        params = {}
        if limit:
            params["limit"] = limit
        
        response = self._request("GET", f"/session/{session_id}/message", params=params, log_request=log_request)
        return response.json()
    
    def get_session(self, session_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/session/{session_id}")
        return response.json()
    
    def send_message_async(
        self,
        session_id: str,
        prompt: str,
        model: Optional[dict[str, str]] = None,
        tools: Optional[dict[str, bool]] = None,
        system: Optional[str] = None,
        agent: Optional[str] = None
    ) -> None:
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": prompt}]
        }
        
        if model:
            body["model"] = model
        
        if tools:
            body["tools"] = tools
        
        if system:
            body["system"] = system
        
        if agent:
            body["agent"] = agent
        
        logger.debug(f"Sending async message to session {session_id} (prompt length: {len(prompt)})")
        
        try:
            response = self._request("POST", f"/session/{session_id}/prompt_async", json=body)
            if response.status_code != 204:
                logger.warning(f"Unexpected status code from prompt_async: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send async message to session {session_id}: {e}")
            raise
    
    def get_session_status(self) -> dict[str, str]:
        response = self._request("GET", "/session/status", log_request=False)
        return response.json()
    
    def abort_session(self, session_id: str) -> bool:
        logger.info(f"Aborting session {session_id}")
        response = self._request("POST", f"/session/{session_id}/abort")
        return response.json()
    
    def delete_session(self, session_id: str) -> bool:
        logger.info(f"Deleting session {session_id}")
        response = self._request("DELETE", f"/session/{session_id}")
        return response.json()
    
    def get_mcp_status(self) -> dict[str, Any]:
        response = self._request("GET", "/mcp")
        return response.json()
    
    def get_session_export(self, session_id: str) -> dict[str, Any]:
        logger.info(f"Exporting session {session_id} for cost data")
        response = self._request("GET", f"/session/{session_id}/export")
        return response.json()
    
    def get_stats(self) -> dict[str, Any]:
        response = self._request("GET", "/stats")
        return response.json()
    
    def get_openapi_spec(self) -> dict[str, Any]:
        response = self._request("GET", "/openapi.json")
        return response.json()
    
    def close(self):
        logger.info("Closing OpenCodeClient")
        self.client.close()


class MessageParser:
    
    @staticmethod
    def extract_text_content(message: dict[str, Any]) -> list[str]:
        texts = []
        
        parts = message.get("parts", [])
        
        for part in parts:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in ["text", "reasoning"]:
                    text = part.get("text", "")
                    if text:
                        texts.append(text)
        
        return texts
    
    @staticmethod
    def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        tool_calls = []
        parts = message.get("parts", [])
        
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "tool":
                tool_info = part.get("tool")
                
                if isinstance(tool_info, dict):
                    tool_calls.append({
                        "id": part.get("callID"),
                        "name": tool_info.get("name"),
                        "input": tool_info.get("arguments", {})
                    })
                elif isinstance(tool_info, str):
                    tool_calls.append({
                        "id": part.get("callID"),
                        "name": tool_info,
                        "input": {}
                    })
        
        return tool_calls
    
    @staticmethod
    def extract_tool_results(message: dict[str, Any]) -> list[dict[str, Any]]:
        tool_results = []
        parts = message.get("parts", [])
        
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                tool_results.append({
                    "tool_use_id": part.get("tool_use_id"),
                    "content": part.get("content", []),
                    "is_error": part.get("is_error", False)
                })
        
        return tool_results
    
    @staticmethod
    def get_message_role(message: dict[str, Any]) -> str:
        info = message.get("info", {})
        return info.get("role", "unknown")
    
    @staticmethod
    def extract_usage_info(message: dict[str, Any]) -> Optional[dict[str, Any]]:
        info = message.get("info", {})
        
        cost = info.get("cost")
        tokens = info.get("tokens")
        
        role = info.get("role")
        if role == "assistant":
            logger.info(f"Assistant message info keys: {list(info.keys())}")
            if cost is not None:
                logger.info(f"Found cost: {cost}")
            if tokens is not None:
                logger.info(f"Found tokens: {tokens}")
        
        if cost is None and tokens is None:
            return None
        
        if tokens:
            return {
                "prompt_tokens": tokens.get("input", 0),
                "completion_tokens": tokens.get("output", 0),
                "total_tokens": tokens.get("total", 0),
                "cost": cost if cost is not None else 0.0,
                "cached_tokens": tokens.get("cache", {}).get("read", 0),
                "reasoning_tokens": tokens.get("reasoning", 0)
            }
        
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": cost if cost is not None else 0.0,
            "cached_tokens": 0,
            "reasoning_tokens": 0
        }
    
    @staticmethod
    def parse_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
        parsed = {
            "reasoning": [],
            "tool_calls": [],
            "tool_results": [],
            "errors": [],
            "usage": None,
            "total_cost": 0.0
        }
        
        if not messages:
            logger.warning("parse_messages received empty message list")
            return parsed
        
        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"Skipping non-dict message: {type(msg)}")
                continue
                
            role = MessageParser.get_message_role(msg)
            
            if role == "assistant":
                texts = MessageParser.extract_text_content(msg)
                parsed["reasoning"].extend(texts)
                
                tool_calls = MessageParser.extract_tool_calls(msg)
                parsed["tool_calls"].extend(tool_calls)
                
                usage_info = MessageParser.extract_usage_info(msg)
                if usage_info:
                    parsed["usage"] = usage_info
                    parsed["total_cost"] += usage_info.get("cost", 0.0)
            
            elif role == "user":
                tool_results = MessageParser.extract_tool_results(msg)
                parsed["tool_results"].extend(tool_results)
        
        return parsed
