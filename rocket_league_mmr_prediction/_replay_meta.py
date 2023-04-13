import abc
import datetime
import logging
import itertools


logger = logging.getLogger(__name__)


class UnknownPlatform(Exception):

    def __init__(self, platform):
        self.platform = platform


class _PlatformPlayerType(abc.ABCMeta):
    def __init__(self, name, bases, attrs):
        if self.platform is not None:
            self.name_to_class[self.platform] = self
        for header_platform_name in self.header_platform_names:
            self.header_name_to_class[header_platform_name] = self
        for ballchasing_platform_name in self.ballchasing_platform_names:
            self.ballchasing_name_to_class[ballchasing_platform_name] = self
        for remote_id_name in self.remote_id_platform_names:
            self.remote_id_name_to_class[remote_id_name] = self
        super().__init__(name, bases, attrs)


class PlatformPlayer(abc.ABC, metaclass=_PlatformPlayerType):
    """Object representing a rocket league player."""
    type_attribute = "__platform_player_type__"
    name_to_class = {}
    header_name_to_class = {}
    ballchasing_name_to_class = {}
    remote_id_name_to_class = {}

    platform = None
    header_platform_names = []
    ballchasing_platform_names = []
    remote_id_platform_names = []

    def __init__(self, display_name, online_id=None):
        self._display_name = display_name
        self._online_id = online_id

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
        return self.name_to_class[obj[self.type_attribute]].from_dict(obj)

    @classmethod
    def from_carball_player(cls, player):
        try:
            class_type = cls.header_name_to_class[player.platform['value']]
        except (AttributeError, KeyError):
            return None
        return class_type.from_carball_player(player)

    @classmethod
    def from_ballchasing_player(cls, player: dict):
        class_type = cls.ballchasing_name_to_class[player["id"]["platform"]]
        return class_type.from_ballchasing_player(player)

    @classmethod
    def from_header_stats(cls, header_stats: dict):
        try:
            platform = header_stats['Platform']['value']
        except:
            import ipdb; ipdb.set_trace()
        try:
            class_type = cls.header_name_to_class[platform]
        except KeyError:
            raise UnknownPlatform(platform)
        return class_type.from_header_stats(header_stats)

    @classmethod
    def from_boxcar_frames_player_info(cls, info):
        try:
            platform_name = next(iter(info['remote_id'].keys()))
        except (KeyError, StopIteration):
            raise
        else:
            class_type = cls.remote_id_name_to_class[platform_name]
            return class_type.from_boxcar_frames_player_info(info)
        return cls.from_header_stats(info)

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
    remote_id_platform_names = ["Steam"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._online_id is None:
            import ipdb; ipdb.set_trace()

    @property
    def tracker_identifier(self):
        return self._online_id

    @property
    def name(self):
        return self._display_name

    @classmethod
    def from_dict(cls, obj):
        return cls(obj["display_name"], online_id=obj.get("online_id", obj.get("identifier")))

    def to_dict(self):
        return {
            self.type_attribute: self.platform,
            "display_name": self._display_name,
            "online_id": self._online_id
        }

    def __eq__(self, other):
        return self._online_id == other._online_id and self.platform == other.platform

    @classmethod
    def from_ballchasing_player(cls, player):
        return cls(player["name"], player["id"]["id"])

    @classmethod
    def from_carball_player(cls, player):
        return cls(player.name, player.online_id)

    @classmethod
    def from_header_stats(cls, header_stats: dict):
        return cls(header_stats['Name'], online_id=header_stats['OnlineID'])

    @classmethod
    def from_boxcar_frames_player_info(cls, info):
        return cls(info['name'], online_id=info['remote_id']['Steam'])


class _DisplayNameSuffixPlayer(PlatformPlayer):

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

    @classmethod
    def from_boxcar_frames_player_info(cls, info):
        return cls(info['name'])


class EpicPlayer(_DisplayNameSuffixPlayer):

    platform = "epic"
    header_platform_names = ['OnlinePlatform_Epic']
    ballchasing_platform_names = ["epic"]
    remote_id_platform_names = ["Epic"]


class PsnPlayer(_DisplayNameSuffixPlayer):

    platform = "psn"
    header_platform_names = ['OnlinePlatform_PS4']
    ballchasing_platform_names = ["ps4", "psynet"]
    remote_id_platform_names = ["PlayStation", "PsyNet"]


class XboxPlayer(_DisplayNameSuffixPlayer):

    platform = "xbl"
    header_platform_names = ['OnlinePlatform_Dingo']
    ballchasing_platform_names = ["xbox"]
    remote_id_platform_names = ["Xbox"]


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
            [PlatformPlayer.from_boxcar_frames_player_info(p) for p in meta['team_zero']],
            [PlatformPlayer.from_boxcar_frames_player_info(p) for p in meta['team_one']],
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
        return itertools.chain(self.team_zero, self.team_one)
