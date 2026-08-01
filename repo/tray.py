import functools


def memoize(fn):
    """Cache a nullary method."""
    return functools.cache(fn)


class Tray:
    """A tray of widgets."""

    @property
    def widgets(self):
        return self._widgets

    @memoize
    def count(self):
        return len(self._widgets)
