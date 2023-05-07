import enum


class Playlist(enum.StrEnum):
    DUEL = "Ranked Duel 1v1"
    DOUBLES = "Ranked Doubles 2v2"
    STANDARD = "Ranked Standard 3v3"

    @classmethod
    def from_string_or_number(cls, number_or_string):
        try:
            return number_to_playlist[int(number_or_string)]
        except Exception:
            return cls(number_or_string)

    @property
    def player_count(self):
        return playlist_to_player_count[self]

    @property
    def ballchasing_filter_string(self):
        return '-'.join(self.split(' ')[:-1]).lower().replace('duel', 'duels')


number_playlist_pairs = list(enumerate(list(Playlist), start=1))
number_to_playlist = dict(number_playlist_pairs)
playlist_to_players_per_team = {
    playlist: count for count, playlist in number_playlist_pairs
}
playlist_to_player_count = {
    playlist: count * 2 for count, playlist in number_playlist_pairs
}
