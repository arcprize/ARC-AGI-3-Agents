from __future__ import annotations

import json
import logging
import os
from threading import Thread
from typing import TYPE_CHECKING, Optional, Type

import requests

from .structs import Scorecard

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger()


class Swarm:
    """Orchestration for many agents playing many ARC-AGI-3 games."""

    GAMES: list[str]
    ROOT_URL: str
    COUNT: int
    agent_name: str
    agent_class: Type[Agent]
    threads: list[Thread]
    agents: list[Agent]
    record_games: list[str]
    cleanup_threads: list[Thread]
    headers: dict[str, str]
    card_id: Optional[str]
    _session: requests.Session

    def __init__(
        self,
        agent: str,
        ROOT_URL: str,
        games: list[str],
        tags: list[str] = [],
    ) -> None:
        from . import AVAILABLE_AGENTS

        self.GAMES = games
        self.ROOT_URL = ROOT_URL
        self.agent_name = agent
        self.agent_class = AVAILABLE_AGENTS[agent]
        self.threads = []
        self.agents = []
        self.cleanup_threads = []
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        self.tags = tags.copy() if tags else []

        # Set up base tags for tracing
        if self.agent_name.endswith(".recording.jsonl"):
            # Extract GUID from playback filename
            # Format: game.agent.count.guid.recording.jsonl
            parts = self.agent_name.split(".")
            guid = parts[-3] if len(parts) >= 4 else "unknown"
            self.tags.extend(["playback", guid])
        else:
            self.tags.extend(["agent", self.agent_name])

    def main(self) -> Scorecard:
        """The main orchestration loop, continues until all agents are done."""

        # submit start of scorecard
        self.card_id = self.open_scorecard()

        # create all the agents
        for i in range(len(self.GAMES)):
            g = self.GAMES[i % len(self.GAMES)]
            a = self.agent_class(
                card_id=self.card_id,
                game_id=g,
                agent_name=self.agent_name,
                ROOT_URL=self.ROOT_URL,
                record=True,
                tags=self.tags,
            )
            self.agents.append(a)

        # create all the threads
        for a in self.agents:
            self.threads.append(Thread(target=a.main, daemon=True))

        # start all the threads
        for t in self.threads:
            t.start()

        # wait for all agent to finish
        for t in self.threads:
            t.join()

        # all agents are now done
        card_id_for_url = self.card_id
        scorecard = self.close_scorecard(self.card_id)
        logger.info("--- FINAL SCORECARD REPORT ---")
        logger.info(json.dumps(scorecard.model_dump(), indent=2))

        # Provide web link to scorecard
        if card_id_for_url:
            scorecard_url = f"{self.ROOT_URL}/scorecards/{card_id_for_url}"
            logger.info(f"View your scorecard online: {scorecard_url}")

        self.cleanup(scorecard)

        return scorecard

    def open_scorecard(self) -> str:
        json_str = json.dumps({"tags": self.tags})

        r = self._session.post(
            f"{self.ROOT_URL}/api/scorecard/open",
            json=json.loads(json_str),
            headers=self.headers,
        )
        
        # Check if request was successful
        if not r.ok:
            logger.error(f"Failed to open scorecard: {r.status_code} - {r.text}")
            raise Exception(f"Failed to open scorecard: {r.status_code}")
            
        response_data = r.json()
        logger.info(f"Response data: {response_data}")
        if "error" in response_data:
            logger.error(f"API error during open scorecard: {response_data}")
            raise Exception(f"API error: {response_data['error']}")
            
        if "card_id" not in response_data:
            logger.error(f"Missing card_id in response: {response_data}")
            raise Exception("Missing card_id in response")
            
        return str(response_data["card_id"])

    def close_scorecard(self, card_id: str) -> Scorecard:
        self.card_id = None
        json_str = json.dumps({"card_id": card_id})
        r = self._session.post(
            f"{self.ROOT_URL}/api/scorecard/close",
            json=json.loads(json_str),
            headers=self.headers,
        )
        
        # Check if request was successful
        if not r.ok:
            logger.error(f"Failed to close scorecard: {r.status_code} - {r.text}")
            raise Exception(f"Failed to close scorecard: {r.status_code}")
            
        response_data = r.json()
        if "error" in response_data:
            logger.error(f"API error during close scorecard: {response_data}")
            raise Exception(f"API error: {response_data['error']}")
            
        return Scorecard.model_validate(response_data)

    def cleanup(self, scorecard: Optional[Scorecard] = None) -> None:
        """Cleanup all agents."""
        for a in self.agents:
            a.cleanup(scorecard)
        if hasattr(self, "_session"):
            self._session.close()
