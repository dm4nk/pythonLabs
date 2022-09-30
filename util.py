
def check_array_length(array: []) -> bool:
    return len(array) == 3


def check_if_int(number: str) -> bool:
    try:
        int(number)
        return True
    except:
        return False


def is_line_correct(line: [str]) -> bool:
    return line and check_array_length(line) and check_if_int(line[0])