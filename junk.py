# go through the files in mock_outputs and change the filenames by prepending german_ and appending _mockoutputs.txt
import os
for filename in os.listdir("mock_outputs"):
    if filename.endswith(".txt"):
        new_filename = "german_" + filename.rsplit("_output.txt", 1)[0] + "_mockoutputs.txt"
        os.rename(os.path.join("mock_outputs", filename), os.path.join("mock_outputs", new_filename))