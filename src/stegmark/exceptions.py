from __future__ import annotations


class StegMarkError(Exception):
    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class InvalidInputError(StegMarkError):
    pass


class MessageTooLongError(InvalidInputError):
    pass


class UnsupportedFormatError(InvalidInputError):
    pass


class ImageTooSmallError(InvalidInputError):
    pass


class FileError(StegMarkError):
    pass


class ImageReadError(FileError):
    pass


class ImageWriteError(FileError):
    pass


class ConfigError(StegMarkError):
    pass


class EngineUnavailableError(StegMarkError):
    pass


class WatermarkError(StegMarkError):
    pass


class WatermarkNotFoundError(WatermarkError):
    pass
