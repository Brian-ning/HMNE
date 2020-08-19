# -*- coding: utf-8 -*-
import AMNE  # The method of import local Python file
import Baselines


# Define the parameter class
class Parakeyward:
    """ Parameter Class """
    def __init__(self, path, sample_p, p, q, num_walks, walks_length, r, examnum, dimensions, workers):
        self.path = path  # The file of path
        self.sample_p = sample_p  # the sample proportion for testing alogrithm
        self.p2return = p  # the parameter of return
        self.q2return = q  # the parameter of In-Out
        self.num_walks = num_walks  # the windows size of Skip-gram
        self.walks_length = walks_length  # the length of walk, like as the sequence of word
        self.radio = r  # the jump propotation of interlayers
        self.examnum = examnum
        self.dimensions = dimensions
        self.workers = workers


# The main function for code
if __name__ == "__main__":
    file = ['CKM'] # "CKM", "ArXiv"
    dimension = 128
    datasets_len = len(file)
    sampling = 0.8
    run_flag = 'train'#'compare'#'train'#'test'#
    for i in range(datasets_len):
        print("------------------%s------------------"%file[i])
        path = "pickle/" + file[i]
        if run_flag == "train":
            train_amne = AMNE.train_model(path, sampling, dimension)
            train_amne.train_AMNE()
        elif run_flag == "test":
            train_amne = AMNE.train_model(path, sampling, dimension)
            train_amne.test_modal()
        elif run_flag == "compare":
            baselines = Baselines.run(path, dimension)
            baselines.baselines_run()
        else:
            print("Input run_flag is error!")

