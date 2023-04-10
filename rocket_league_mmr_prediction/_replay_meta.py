import abc
import datetime
import logging

from carball_lite.player import Player as CarballPlayer
from carball_lite.game import Game as CarballGame


logger = logging.getLogger(__name__)


class _PlatformPlayerType(abc.ABCMeta):
    def __init__(self, name, bases, attrs):
        if self.platform is not None:
            self.type_to_class[self.platform] = self
        for carball_platform_name in self.carball_platform_names:
            self.carball_type_to_class[carball_platform_name] = self

        for ballchasing_platform_name in self.ballchasing_platform_names:
            self.ballchasing_type_to_class[ballchasing_platform_name] = self
        super().__init__(name, bases, attrs)


class PlatformPlayer(abc.ABC, metaclass=_PlatformPlayerType):
    """Object representing a rocket league player."""
    type_attribute = "__platform_player_type__"
    type_to_class = {}
    carball_type_to_class = {}
    ballchasing_type_to_class = {}

    platform = None
    carball_platform_names = []
    ballchasing_platform_names = []

    @abc.abstractproperty
    def tracker_identifier(self):
        """The identifier to use to look up the players profile on the tracker network."""
        pass

    @abc.abstractproperty
    def name(self):
        """The name of the player."""
        pass

    @property
    def tracker_suffix(self):
        """The url suffix that should be used to find the players profile on the tracker network."""
        return f"{self.platform}/{self.tracker_identifier}"

    @abc.abstractmethod
    def to_dict(self):
        pass

    @classmethod
    def from_dict(self, obj):
        return self.type_to_class[obj[self.type_attribute]].from_dict(obj)

    @classmethod
    def from_carball_player(cls, player: CarballPlayer):
        try:
            class_type = cls.carball_type_to_class[player.platform['value']]
        except AttributeError:
            import ipdb; ipdb.set_trace()
        return class_type.from_carball_player(player)

    @classmethod
    def from_ballchasing_player(cls, player: dict):
        class_type = cls.ballchasing_type_to_class[player["id"]["platform"]]
        return class_type.from_ballchasing_player(player)

    def matches_carball(self, player: CarballPlayer):
        try:
            return self.from_carball_player(player) == self
        except Exception as e:
            logger.warn(f"Using imperfect equality for {player} because {e}")
            return player.name == self.name

    def __eq__(self, other):
        return self.name == other.name and self.platform == other.platform


class SteamPlayer(PlatformPlayer):
    """A player on the steam platform."""

    platform = "steam"
    carball_platform_names = ['OnlinePlatform_Steam']
    ballchasing_platform_names = ["steam"]

    def __init__(self, display_name, identifier):
        self._display_name = display_name
        self._identifier = identifier

    @property
    def tracker_identifier(self):
        return self._identifier

    @property
    def name(self):
        return self._display_name

    @classmethod
    def from_dict(cls, obj):
        return cls(obj["display_name"], obj["identifier"])

    def to_dict(self):
        return {
            self.type_attribute: self.platform,
            "display_name": self._display_name,
            "identifier": self._identifier
        }

    def __eq__(self, other):
        return self._identifier == other._identifier and self.platform == other.platform

    @classmethod
    def from_ballchasing_player(cls, player):
        return cls(player["name"], player["id"]["id"])

    @classmethod
    def from_carball_player(cls, player: CarballPlayer):
        return cls(player.name, player.online_id)


class _DisplayNameSuffixPlayer(PlatformPlayer):

    def __init__(self, display_name):
        self._display_name = display_name

    @property
    def tracker_identifier(self):
        return self._display_name.replace(" ", "%20")

    @property
    def name(self):
        return self._display_name

    @classmethod
    def from_dict(cls, obj):
        return cls(obj["display_name"])

    def to_dict(self):
        return {
            self.type_attribute: self.platform,
            "display_name": self._display_name
        }

    @classmethod
    def from_carball_player(cls, player: CarballPlayer):
        return cls(player.name)

    @classmethod
    def from_ballchasing_player(cls, player):
        return cls(player["name"])


class EpicPlayer(_DisplayNameSuffixPlayer):

    platform = "epic"
    carball_platform_names = ['OnlinePlatform_Epic']
    ballchasing_platform_names = ["epic"]


class PsnPlayer(_DisplayNameSuffixPlayer):

    platform = "psn"
    carball_platform_names = ['OnlinePlatform_PS4']
    ballchasing_platform_names = ["ps4"]


class XboxPlayer(_DisplayNameSuffixPlayer):

    platform = "xbl"
    carball_platform_names = ['OnlinePlatform_Dingo']
    ballchasing_platform_names = ["xbox"]


class ReplayMeta:
    """Metadata associated with a rocket league replay"""

    @classmethod
    def from_dict(cls, obj):
        """Create a :py:class:`ReplayMeta` from a dictionary."""
        return cls(
            datetime.datetime.fromisoformat(obj["datetime"]),
            [PlatformPlayer.from_dict(player) for player in obj["orange_players"]],
            [PlatformPlayer.from_dict(player) for player in obj["blue_players"]]
        )

    @classmethod
    def from_carball_game(cls, game: CarballGame):
        """Create a :py:class:`ReplayMeta` from a :py:class:`CarballGame`."""
        game_datetime = datetime.datetime.strptime(game.properties['Date'], "%Y-%m-%d %H-%M-%S")
        orange_team = list(
            next(team.players for team in game.teams if team.is_orange)
        )
        blue_team = list(
            next(team.players for team in game.teams if not team.is_orange)
        )
        return cls(
            game_datetime,
            [
                PlatformPlayer.from_carball_player(player)
                for player in orange_team
            ],
            [
                PlatformPlayer.from_carball_player(player)
                for player in blue_team
            ]
        )

    @classmethod
    def from_ballchasing_game(cls, game: dict):
        """Create a :py:class:`ReplayMeta` from a ballchasing game."""
        game_datetime = datetime.datetime.fromisoformat(game["date"])
        orange_team = game["orange"]["players"]
        blue_team = game["blue"]["players"]
        return cls(
            game_datetime,
            [
                PlatformPlayer.from_ballchasing_player(player)
                for player in orange_team
            ],
            [
                PlatformPlayer.from_ballchasing_player(player)
                for player in blue_team
            ]
        )

    def __init__(
            self, replay_datetime: datetime.datetime,
            orange_order: [PlatformPlayer], blue_order: [PlatformPlayer]
    ):
        self.datetime = replay_datetime
        self.orange_players = orange_order
        self.blue_players = blue_order

    def to_dict(self):
        return {
            "datetime": self.datetime.isoformat(),
            "orange_players": [player.to_dict() for player in self.orange_players],
            "blue_players": [player.to_dict() for player in self.blue_players]
        }

    @property
    def player_order(self):
        self.orange_players.sort(key=lambda p: p.name)
        self.blue_players.sort(key=lambda p: p.name)
        return self.orange_players + self.blue_players
