text = open("data/train.txt", encoding="utf-8").read()
open("data/train.txt", "w", encoding="utf-8").write(text * 50)
print("Done!")