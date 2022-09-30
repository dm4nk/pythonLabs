from util import is_line_correct


class Record:
    def __init__(self):
        self.__number_courses: int
        self.__time_study: float
        self.__marks: float

    def __init__(self, number_courses: int, time_study: float, marks: float):
        self.__number_courses: int = number_courses
        self.__time_study: float = time_study
        self.__marks: float = marks

    @property
    def number_courses(self) -> int:
        return self.__number_courses

    @property
    def time_study(self) -> float:
        return self.__time_study

    @property
    def marks(self) -> float:
        return self.__marks

    @number_courses.setter
    def number_courses(self, number_courses: int) -> None:
        if number_courses < 0:
            raise Exception("Like cannot be below zero")
        self.__number_courses = number_courses

    @time_study.setter
    def time_study(self, time_study: float) -> None:
        self.__time_study = time_study

    @marks.setter
    def marks(self, marks: float) -> None:
        self.__marks = marks

    def __str__(self):
        return f'Record({self.number_courses}, {self.time_study}, {self.marks})'


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

    def sort_by_marks(self):
        self.records = sorted(self.__records, key=lambda r: r.marks)

    def sort_by_number_courses(self):
        self.records = sorted(self.__records, key=lambda r: r.number_courses)

    def sort_by_time_study(self):
        self.records = sorted(self.__records, key=lambda r: r.marks)

    def load(self, file: str):
        records = []
        with open(file) as f:
            data = f.readlines()
            for line in data:
                line = line.strip()
                line = line.split(',')

                if not is_line_correct(line):
                    continue

                records.append(Record(int(line[0]), float(line[1]), float(line[2])))
        self.__records = records

    def __str__(self):
        return f'{[str(record) for record in self.__records]}'
