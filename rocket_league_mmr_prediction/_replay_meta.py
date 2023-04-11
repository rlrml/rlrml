import abc
import datetime
import logging


logger = logging.getLogger(__name__)


class UnknownPlatform(Exception):

    def __init__(self, platform):
        self.platform = platform


class _PlatformPlayerType(abc.ABCMeta):
    def __init__(self, name, bases, attrs):
        if self.platform is not None:
            self.type_to_class[self.platform] = self
        for header_platform_name in self.header_platform_names:
            self.header_type_to_class[header_platform_name] = self

        for ballchasing_platform_name in self.ballchasing_platform_names:
            self.ballchasing_type_to_class[ballchasing_platform_name] = self
        super().__init__(name, bases, attrs)


class PlatformPlayer(abc.ABC, metaclass=_PlatformPlayerType):
    """Object representing a rocket league player."""
    type_attribute = "__platform_player_type__"
    type_to_class = {}
    header_type_to_class = {}
    ballchasing_type_to_class = {}

    platform = None
    header_platform_names = []
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
    def from_carball_player(cls, player):
        try:
            class_type = cls.header_type_to_class[player.platform['value']]
        except (AttributeError, KeyError):
            return None
        return class_type.from_carball_player(player)

    @classmethod
    def from_ballchasing_player(cls, player: dict):
        class_type = cls.ballchasing_type_to_class[player["id"]["platform"]]
        return class_type.from_ballchasing_player(player)

    @classmethod
    def from_header_stats(cls, header_stats: dict):
        platform = header_stats['Platform']['value']
        try:
            class_type = cls.header_type_to_class[platform]
        except KeyError:
            raise UnknownPlatform(platform)
        return class_type.from_header_stats(header_stats)

    def matches_carball(self, player):
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
    header_platform_names = ['OnlinePlatform_Steam']
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
    def from_carball_player(cls, player):
        return cls(player.name, player.online_id)

    @classmethod
    def from_header_stats(cls, header_stats: dict):
        return cls(header_stats['Name'], header_stats['OnlineID'])


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
    def from_carball_player(cls, player):
        return cls(player.name)

    @classmethod
    def from_ballchasing_player(cls, player):
        return cls(player["name"])

    @classmethod
    def from_header_stats(cls, header_stats: dict):
        return cls(header_stats['Name'])


class EpicPlayer(_DisplayNameSuffixPlayer):

    platform = "epic"
    header_platform_names = ['OnlinePlatform_Epic']
    ballchasing_platform_names = ["epic"]


class PsnPlayer(_DisplayNameSuffixPlayer):

    platform = "psn"
    header_platform_names = ['OnlinePlatform_PS4']
    ballchasing_platform_names = ["ps4"]


class XboxPlayer(_DisplayNameSuffixPlayer):

    platform = "xbl"
    header_platform_names = ['OnlinePlatform_Dingo']
    ballchasing_platform_names = ["xbox"]


class ReplayMeta:
    """Metadata associated with a rocket league replay"""

    @classmethod
    def from_dict(cls, obj):
        """Create a :py:class:`ReplayMeta` from a dictionary."""
        return cls(
            datetime.datetime.fromisoformat(obj["datetime"]),
            [PlatformPlayer.from_dict(player) for player in obj["team_zero"]],
            [PlatformPlayer.from_dict(player) for player in obj["team_one"]]
        )

    @classmethod
    def from_boxcar_frames_meta(cls, meta):
        # players should already be ordered with team 0 coming first and team 1 second
        headers = dict(meta['all_headers'])
        return cls(
            datetime.datetime.strptime(headers['Date'], "%Y-%m-%d %H-%M-%S"),
            [PlatformPlayer.from_header_stats(p['stats']) for p in meta['team_zero']],
            [PlatformPlayer.from_header_stats(p['stats']) for p in meta['team_one']],
        )

    @classmethod
    def from_carball_game(cls, game):
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
            team_zero: [PlatformPlayer], team_one: [PlatformPlayer]
    ):
        self.datetime = replay_datetime
        self.team_zero = team_zero
        self.team_one = team_one

    def to_dict(self):
        return {
            "datetime": self.datetime.isoformat(),
            "team_zero": [player.to_dict() for player in self.team_zero],
            "team_one": [player.to_dict() for player in self.team_one]
        }

    @property
    def player_order(self):
        return self.team_zero + self.team_one
