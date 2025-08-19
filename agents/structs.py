import json
from enum import Enum
from typing import Any, Optional, Type, Union

from pydantic import BaseModel, Field, computed_field, field_validator

MAX_REASONING_BYTES = 16 * 1024  # 16KB Max


class GameState(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class Card(BaseModel):
    """
    A single scorecard for a single game. A game can be played more than
    once, we track each play with lists of card properties (scores, states, actions)
    """

    game_id: str
    total_plays: int = 0

    guids: list[str] = Field(default_factory=list, exclude=True)
    scores: list[int] = Field(default_factory=list)
    states: list[GameState] = Field(default_factory=list)
    actions: list[int] = Field(default_factory=list)
    resets: list[int] = Field(default_factory=list)

    @property
    def idx(self) -> int:
        # lists are zero indexed by play_count starts at 1
        return self.total_plays - 1

    @property
    def started(self) -> bool:
        return self.total_plays > 0

    @property
    def score(self) -> Optional[int]:
        return self.scores[self.idx] if self.started else None

    @property
    def high_score(self) -> int:
        return max(self.scores) if self.started else 0

    @property
    def state(self) -> str:
        return self.states[self.idx] if self.started else GameState.NOT_PLAYED

    @property
    def action_count(self) -> Optional[int]:
        return self.actions[self.idx] if self.started else None

    @property
    def total_actions(self) -> int:
        return sum(self.actions)


class Scorecard(BaseModel):
    """
    Tracks and holds the scorecard for all games
    """

    games: list[str] = Field(default_factory=list, exclude=True)
    cards: dict[str, Card] = Field(default_factory=dict)
    source_url: Optional[str] = None
    tags: Optional[list[str]] = None
    opaque: Optional[Any] = Field(default=None)
    card_id: str = ""
    api_key: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.cards:
            self.cards = {}

    @computed_field(return_type=int)
    def won(self) -> int:
        return sum(GameState.WIN in g.states for g in self.cards.values())

    @computed_field(return_type=int)
    def played(self) -> int:
        return sum(bool(g.states) for g in self.cards.values())

    @computed_field(return_type=int)
    def total_actions(self) -> int:
        return sum(g.total_actions for g in self.cards.values())

    @computed_field(return_type=int)
    def score(self) -> int:
        return sum(g.high_score for g in self.cards.values())

    def get(self, game_id: Optional[str] = None) -> dict[str, Any]:
        if game_id is not None:
            card = self.cards.get(game_id)
            return {game_id: card.model_dump()} if card else {}
        return {k: v.model_dump() for k, v in self.cards.items()}

    def get_json_for(self, game_id: str) -> dict[str, Any]:
        card = self.cards.get(game_id)
        return {
            "won": self.won,
            "played": self.played,
            "total_actions": self.total_actions,
            "score": self.score,
            "cards": {game_id: card.model_dump()} if card else {},
        }


class SimpleAction(BaseModel):
    game_id: str = ""


class ComplexAction(BaseModel):
    game_id: str = ""
    x: int = Field(default=0, ge=0, le=63)
    y: int = Field(default=0, ge=0, le=63)


class GameAction(Enum):
    RESET = (0, SimpleAction)
    ACTION1 = (1, SimpleAction)
    ACTION2 = (2, SimpleAction)
    ACTION3 = (3, SimpleAction)
    ACTION4 = (4, SimpleAction)
    ACTION5 = (5, SimpleAction)
    ACTION6 = (6, ComplexAction)
    ACTION7 = (7, SimpleAction)

    action_type: Union[Type[SimpleAction], Type[ComplexAction]]
    action_data: Union[SimpleAction, ComplexAction]
    reasoning: Optional[Any]

    def __init__(
        self,
        action_id: int,
        action_type: Union[Type[SimpleAction], Type[ComplexAction]],
    ) -> None:
        self._value_ = action_id
        self.action_type = action_type
        self.action_data = action_type()
        self.reasoning = None

    def is_simple(self) -> bool:
        return self.action_type is SimpleAction

    def is_complex(self) -> bool:
        return self.action_type is ComplexAction

    def validate_data(self, data: dict[str, Any]) -> bool:
        """Raise exception on invalid parse of incoming JSON data."""
        self.action_type.model_validate(data)
        return True

    def set_data(self, data: dict[str, Any]) -> Union[SimpleAction, ComplexAction]:
        self.action_data = self.action_type(**data)
        return self.action_data

    @classmethod
    def from_id(cls, action_id: int) -> "GameAction":
        for action in cls:
            if action.value == action_id:
                return action
        raise ValueError(f"No GameAction with id {action_id}")

    @classmethod
    def from_name(cls, name: str) -> "GameAction":
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"No GameAction with name '{name}'")

    @classmethod
    def all_simple(cls) -> list["GameAction"]:
        return [a for a in cls if a.is_simple()]

    @classmethod
    def all_complex(cls) -> list["GameAction"]:
        return [a for a in cls if a.is_complex()]


class ActionInput(BaseModel):
    id: GameAction = GameAction.RESET
    data: dict[str, Any] = {}
    reasoning: Optional[Any] = Field(
        default=None,
        description="Opaque client-supplied blob; stored & echoed back verbatim.",
    )

    # Optional size / serialisability guard
    @field_validator("reasoning")
    @classmethod
    def _check_reasoning(cls, v: Any) -> Any:
        if v is None:
            return v  # field omitted → fine
        try:
            raw = json.dumps(v, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError("reasoning must be JSON-serialisable")
        if len(raw) > MAX_REASONING_BYTES:
            raise ValueError(f"reasoning exceeds {MAX_REASONING_BYTES} bytes")
        return v


class FrameData(BaseModel):
    game_id: str = ""
    frame: list[list[list[int]]] = []
    state: GameState = GameState.NOT_PLAYED
    score: int = Field(0, ge=0, le=254)
    action_input: ActionInput = Field(default_factory=lambda: ActionInput())
    guid: Optional[str] = None
    full_reset: bool = False
    available_actions: list[GameAction] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.frame) == 0
