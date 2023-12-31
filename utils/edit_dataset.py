from os import listdir, remove

path_to_dataset = "D:/Prom Detect/Datasets/Dataset 2/"
for file_name in sorted(listdir(path_to_dataset)):
    if not "txt" in file_name:
        continue
    with open(path_to_dataset + file_name, "r") as file:
        if not file.read():
            print(path_to_dataset + file_name)
            remove(path_to_dataset + file_name)
    # if not "txt" in file_name:
    #     continue
    # lines = []
    # with open(path_to_dataset + file_name, "r") as file:
    #     for line in file.read().split("\n"):
    #         try:
    #             if line.split()[0] == "3":
    #                 lines.append(line)
    #         except:
    #             pass

    # with open(path_to_dataset + file_name, "w") as file:
    #     print("\n".join(lines), file=file)