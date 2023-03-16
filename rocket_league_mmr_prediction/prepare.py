"""Tools for the conversion of raw rocket league replays into something that can be used for training and prediction."""
import itertools
import os
from carball_lite.game import Game
from boxcars_py import parse_replay
import numpy as np


def get_carball_game(replay_path):
    """Take a path to the replay and output a Game object associated with that replay.

    :param replay_path: Path to a specific replay.
    :return: The object created from boxcars.
    """
    with open(replay_path, 'rb') as f:
        buf = f.read()
    boxcars_data = parse_replay(buf)
    game = Game()
    # This is confusingly called loaded_json, but what is expected is actually a
    # python object. In our case we are not loading from json, but directly from
    # the replay file, but this is fine.
    game.initialize(loaded_json=boxcars_data)
    return game


def load_games_from_directory(filepath, extension="replay"):
    """Return a numpy array that combines data from every replay file in the provided directory."""
    game_arrays = []
    for _, _, files in os.walk(filepath):
        for filename in files:
            if os.path.splitext(filename)[1] == extension:
                carball_game = get_carball_game(os.path.join(filepath, filename))
                numerical_data = _CarballToNumpyConverter(carball_game).get_numpy_array()
                game_arrays.append(numerical_data)
    return game_arrays


class _CarballToNumpyConverter(object):

    PLAYER_COLUMNS = [
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'rot_x', 'rot_y', 'rot_z',
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    ]

    # These are also available
    # 'ping', 'throttle', 'steer', 'handbrake', 'ball_cam', 'dodge_active',
    # 'boost', 'boost_active', 'jump_active', 'double_jump_active',

    BALL_COLUMNS = [
        'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
    ]

    # These are also available but probably not needed
    EXTRA_BALL_COLUMNS = [
        'rot_x', 'rot_y', 'rot_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no'
    ]

    def __init__(self, carball_game, include_time=True):
        """Initialize the converter."""
        self.carball_game = carball_game
        self.orange_team = next(team for team in carball_game.teams if team.is_orange)
        self.blue_team = next(team for team in carball_game.teams if not team.is_orange)
        self.include_time = include_time

    def get_labels_for_players(self):
        pass

    def get_numpy_array_and_labels(self):
        pass

    def get_numpy_array(self):
        """Return a numpy array from the provided carball game."""
        first_relevant_frame = self._calculate_first_relevant_frame()
        return np.stack([
            self._construct_numpy_frame(i)
            for i in self.carball_game.frames.index
            if i >= first_relevant_frame
        ])

    @classmethod
    def _get_data_frame_value_using_last_as_default(cls, column, carball_frame_index):
        if carball_frame_index < 0:
            # TODO: ..
            raise Exception("Shoudln't happen")
        if carball_frame_index in column.index:
            return column[carball_frame_index]
        # TODO: use a real logger
        print(f"Missing value at {carball_frame_index} for column {column.name}")
        cls._get_data_frame_value_using_last_as_default(column, carball_frame_index - 1)

    def _construct_numpy_frame(self, carball_frame_index):
        ball_values = (
            self._get_data_frame_value_using_last_as_default(
                self.carball_game.ball[column_name], carball_frame_index
            )
            for column_name in self.BALL_COLUMNS
        )
        player_values = (
            self._get_data_frame_value_using_last_as_default(
                player.data[column_name],
                carball_frame_index
            )
            for player in itertools.chain(
                self.orange_team.players, self.blue_team.players
            )
            for column_name in self.PLAYER_COLUMNS
        )
        return np.fromiter(
            itertools.chain(
                [self.carball_game.frames['time'][carball_frame_index]],
                ball_values,
                player_values
            ),
            dtype=float
        )

    def _calculate_first_relevant_frame(self):
        initial_time_remaining = self.carball_game.frames.seconds_remaining.iloc[0]
        for i, v in enumerate(self.carball_game.frames.seconds_remaining):
            if v > initial_time_remaining:
                raise Exception(
                    "There should never be more time remaining than what we started with"
                )
            elif v < initial_time_remaining:
                frame_index = i
                break
        else:
            raise Exception("Time remaining never decreased")

        time_at_most_one_second_in = self.carball_game.frames.time.iloc[frame_index]

        for index in range(frame_index, -1, -1):
            if time_at_most_one_second_in - self.carball_game.frames.time.iloc[index] > 1.01:
                first_relevant_index = index
                break
        else:
            raise Exception("This loop should terminate")

        first_relevant_frame = self.carball_game.frames.index[first_relevant_index]

        return first_relevant_frame
