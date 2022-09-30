import numpy as np

from util import generate_test_shit, squares_sum, square_sums_in_range, build_graph, surface

FILE = 'data/Student_Marks.csv'


def main():
    # rlist = RecordsList()
    # rlist.load(FILE)
    # rlist.records.append(Record(6, 6, 6))
    # rlist.sort_by_marks()
    # rlist.sort_by_time_study()
    # rlist.sort_by_number_courses()
    # print(rlist)

    x, y = generate_test_shit(-4, 0.125, 10, 100)
    k = np.linspace(-400, 400, 100, dtype=float)
    b = np.linspace(-800, 800, 100, dtype=float)
    print(x.tolist(), y.tolist())
    print(squares_sum(x, y, 0.125, -4))
    print(square_sums_in_range(x, y, k, b))
    surface(square_sums_in_range(x, y, k, b))


if __name__ == '__main__':
    main()
