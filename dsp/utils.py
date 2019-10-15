def generate_ampl_code(seq):

    for i, j in enumerate(seq):
        print(f"let seq[{i+1}] := shipo[{j}];")

if __name__ == "__main__":
    seq = [15, 8, 5, 44, 26, 56, 53, 32, 30, 63, 12, 23, 61, 28]
    generate_ampl_code(seq)
