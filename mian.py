from model import RecordsList, Record

FILE = 'testdata.csv'


def main():
    rlist = RecordsList()
    rlist.load(FILE)
    rlist.records.append(Record('ababa', 6))
    rlist.sort_by_likes()
    print(rlist)


if __name__ == '__main__':
    main()
