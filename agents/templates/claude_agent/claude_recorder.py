import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import AssistantMessage, ToolUseBlock, ResultMessage, UserMessage, SystemMessage

logger = logging.getLogger()


class ClaudeCodeRecorder:
    
    def __init__(self, game_id: str, agent_name: str, session_id: Optional[str] = None):
        self.game_id = game_id
        self.agent_name = agent_name
        self.session_id = session_id or str(uuid.uuid4())

        recordings_dir = os.getenv("RECORDINGS_DIR", "recordings")
        self.output_dir = Path(recordings_dir) / f"{game_id}_{agent_name}_{self.session_id}"
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ClaudeCodeRecorder initialized: {self.output_dir}")
        except PermissionError as e:
            logger.error(f"Permission denied creating recording directory {self.output_dir}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create recording directory {self.output_dir}: {e}")
            raise
    
    def save_step(
        self,
        step: int,
        prompt: str,
        messages: list[Any],
        parsed_action: dict[str, Any],
        total_cost_usd: float
    ) -> None:
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            formatted_messages = self.format_messages(messages)
            
            step_data = {
                "step": step,
                "timestamp": timestamp,
                "prompt": prompt,
                "messages": formatted_messages,
                "parsed_action": parsed_action,
                "cost_usd": total_cost_usd
            }
            
            step_filename = self.output_dir / f"step_{step:03d}.json"
            
            if not self.output_dir.exists():
                logger.warning(f"Recording directory {self.output_dir} doesn't exist, recreating")
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(step_filename, "w", encoding="utf-8") as f:
                json.dump(step_data, f, indent=2)
            
            logger.info(f"Saved step {step} to {step_filename}")
        except PermissionError as e:
            logger.error(f"Permission denied writing step {step} to {step_filename}: {e}")
        except IOError as e:
            logger.error(f"IO error saving step {step} to {step_filename}: {e}")
        except Exception as e:
            logger.error(f"Failed to save step {step}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def format_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        formatted = []
        tool_id_to_name = {}
        message_index = 0
        
        for i, msg in enumerate(messages):
            try:
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            formatted.append({
                                "message": message_index,
                                "type": "text",
                                "content": block.text
                            })
                            message_index += 1
                        
                        if isinstance(block, ToolUseBlock):
                            tool_id_to_name[block.id] = block.name
                            formatted.append({
                                "message": message_index,
                                "type": "tool_call",
                                "tool_call_id": block.id,
                                "tool_name": block.name,
                                "tool_input": block.input
                            })
                            message_index += 1
                
                elif isinstance(msg, UserMessage):
                    if hasattr(msg, 'content') and isinstance(msg.content, list):
                        for block in msg.content:
                            if hasattr(block, 'tool_use_id'):
                                content_text = ""
                                if hasattr(block, 'content') and isinstance(block.content, list):
                                    for item in block.content:
                                        if isinstance(item, dict) and 'text' in item:
                                            content_text = item['text']
                                            break
                                
                                formatted.append({
                                    "message": message_index,
                                    "type": "tool_result",
                                    "tool_use_id": block.tool_use_id,
                                    "content": content_text,
                                    "is_error": getattr(block, 'is_error', None)
                                })
                                message_index += 1
                
                elif isinstance(msg, ResultMessage):
                    usage = msg.usage or {}
                    formatted.append({
                        "message": message_index,
                        "type": "result",
                        "duration_ms": msg.duration_ms,
                        "duration_api_ms": msg.duration_api_ms,
                        "is_error": msg.is_error,
                        "tokens": {
                            "input": usage.get("input_tokens", 0),
                            "output": usage.get("output_tokens", 0),
                        }
                    })
                    message_index += 1
            
            except Exception as e:
                logger.warning(f"Error formatting message {i} (type={type(msg).__name__}): {e}")
                formatted.append({
                    "message": message_index,
                    "type": "error",
                    "error": str(e),
                    "msg_type": type(msg).__name__
                })
                message_index += 1
        
        return formatted
    
    def aggregate_responses(self, formatted_messages: list[dict[str, Any]]) -> str:
        text_parts = []
        
        for msg in formatted_messages:
            if msg.get("type") == "text":
                text_parts.append(msg.get("content", ""))
        
        return "".join(text_parts)
