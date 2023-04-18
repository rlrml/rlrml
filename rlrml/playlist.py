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


number_to_playlist = {
    1: Playlist.DUEL,
    2: Playlist.DOUBLES,
    3: Playlist.STANDARD,
}
