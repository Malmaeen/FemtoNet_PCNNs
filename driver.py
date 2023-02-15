if __name__ == "__main__":

    # -- set up the gpu env
    import sys,os
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    # -- import the model and data 
    import tools as tl
    import pcnns as pc
    PCNNs_Model= pc.PCNNS(outdir="saved")
    PCNNs_Model.normalized_data()
    PCNNs_Model.split_train_test()
    ## -- train PCNNs
    #PCNNs_Model.train()
    ## -- Testing 
    PCNNs_Model.plot_result_1()
    PCNNs_Model.plot_result_2()
    
