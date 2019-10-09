def generate_ampl_code(seq):

    for i, j in enumerate(seq):
        print(f"let seq[{i+1}] := shipo[{j}];")

if __name__ == "__main__":
    seq = [33, 15, 8, 26, 56, 53, 5, 44, 32, 30, 38, 12, 63, 4, 23, 61, 28]
    generate_ampl_code(seq)
