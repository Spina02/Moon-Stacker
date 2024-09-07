def progress(current, total, msg):
    print(f'\033[A{current}/{total} {msg}')
    if current == total: print()