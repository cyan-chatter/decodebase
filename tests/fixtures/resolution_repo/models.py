from __future__ import annotations


class Base:
    def load(self) -> None:
        pass

    def save(self) -> None:
        pass


class Repository(Base):
    def __init__(self, conn) -> None:
        self.conn = conn

    def inherited(self) -> None:
        self.load()

    def save(self) -> None:
        self.conn.execute('UPDATE ...')


def save_record() -> None:
    Base().save()
    Repository(None).save()


class Manager:
    def __init__(self) -> None:
        self.repo = Repository(None)

    def commit(self) -> None:
        self.repo.save()
