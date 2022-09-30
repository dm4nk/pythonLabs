from model import RecordsList, Record

FILE = 'data/Student_Marks.csv'


def main():
    rlist = RecordsList()
    rlist.load(FILE)
    rlist.records.append(Record(6, 6, 6))
    rlist.sort_by_marks()
    print(rlist)


if __name__ == '__main__':
    main()
