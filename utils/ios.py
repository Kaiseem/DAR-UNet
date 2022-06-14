import os
def mkdir(r,n):
    p=os.path.join(r,n)
    if not os.path.isdir(p):
        os.makedirs(p)

def create_dirs(name):
    mkdir(name,'i2i_train_visual')
    mkdir(name,'i2i_checkpoints')