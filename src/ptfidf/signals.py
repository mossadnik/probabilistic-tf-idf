"""Base classes for Observer pattern."""

from weakref import WeakSet


class Observable:
    def __init__(self):
        self._observers = WeakSet()

    def subscribe(self, obj):
        self._observers.add(obj)

    def unsubscribe(self, obj):
        self._observers.discard(obj)

    def _notify(self):
        for obj in self._observers:
            obj._receive_update()


class Observer:
    def __init__(self):
        self.uptodate = False

    def _receive_update(self):
        self.uptodate = False

    def update(self):
        self.uptodate = True


def check_uptodate(func):
    def wrapper(*args, **kwargs):
        if not args[0].uptodate:
            args[0].update()
        return func(*args, **kwargs)
    return wrapper
