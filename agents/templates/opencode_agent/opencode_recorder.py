import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger()


class OpenCodeRecorder:
    
    def __init__(self, game_id: str, agent_name: str, session_id: Optional[str] = None):
        self.game_id = game_id
        self.agent_name = agent_name
        self.session_id = session_id or str(uuid.uuid4())

        recordings_dir = os.getenv("RECORDINGS_DIR", "recordings")
        self.output_dir = Path(recordings_dir) / f"{game_id}_{agent_name}_{self.session_id}"
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"OpenCodeRecorder initialized: {self.output_dir}")
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
            
            step_data = {
                "step": step,
                "timestamp": timestamp,
                "prompt": prompt,
                "messages": messages,
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
