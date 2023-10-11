import os

for line in os.listdir('saved_models'):
  if not line.endswith(".txt"):
    continue
  _, filename = line.split(".json")
  filename = "gpt2" + filename
  os.rename("saved_models/" + line, "output/" + filename)
