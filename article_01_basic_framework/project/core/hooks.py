from __future__ import annotations


class Hook:
    def before_epoch(self, state: dict) -> None:
        pass

    def after_epoch(self, state: dict) -> None:
        pass

    def before_step(self, state: dict) -> None:
        pass

    def after_step(self, state: dict) -> None:
        pass


