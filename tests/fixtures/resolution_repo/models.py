from __future__ import annotations


class Base:
    def save(self) -> None:
        pass


class Repository(Base):
    def __init__(self, conn) -> None:
        self.conn = conn

    def save(self) -> None:
        self.conn.execute('UPDATE ...')


def save_record() -> None:
    Base().save()
    Repository(None).save()
