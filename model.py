from util import is_line_correct


class Record:
    def __init__(self):
        self.__text: str
        self.__like: int

    def __init__(self, text: str, like: int):
        self.__text: str = text
        self.__like: int = like

    @property
    def text(self) -> str:
        return self.__text

    @property
    def like(self) -> int:
        return self.__like

    @like.setter
    def like(self, like: int) -> None:
        if like < 0:
            raise Exception("Like cannot be below zero")
        self.__like = like

    @text.setter
    def text(self, text: str) -> None:
        self.__text = text

    def __str__(self):
        return f'{self.text}, {self.__like}'


class RecordsList:
    def __init__(self):
        self.__records: [Record]

    def __int__(self, records: [Record]):
        self.__records: [Record] = records

    @property
    def records(self) -> [Record]:
        return self.__records

    @records.setter
    def records(self, records: [Record]) -> None:
        self.__records = records

    def sort_by_text(self):
        self.records = sorted(self.__records, key=lambda r: r.text)

    def sort_by_likes(self):
        self.records = sorted(self.__records, key=lambda r: r.like)

    def load(self, file: str):
        records = []
        with open(file) as f:
            data = f.readlines()
            for line in data:
                line = line.strip()
                line = line.split(',')

                if not is_line_correct(line):
                    continue

                records.append(Record(line[0], int(line[1])))
        self.__records = records

    def __str__(self):
        return f'{[str(record) for record in self.__records]}'
