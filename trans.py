import os
import shutil
path = "/project/kfold"
data_path = "/project/dataset"
cats = os.listdir(path)
cls=['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
def mkdir(dirs):
    if(os.path.exists(dirs)==False):
        os.mkdir(dirs)
mkdir(os.path.join(data_path,"train"))
mkdir(os.path.join(data_path,"val"))


for cl in cls:
    mkdir(os.path.join(data_path,"train",cl)) 
    mkdir(os.path.join(data_path,"val",cl))     
    datas=[]
    for cat in cats:
        cls_path = os.path.join(path,cat,cl)
        dat = os.listdir(cls_path)

        datas.extend([ os.path.join(cls_path,d)  for d in dat ])
    #print(datas)
    count=0
    for d in datas:

        art = d.split("/")[3]
        fname =  d.split("/")[5]
 
        if((count%6)==0):
            new_dir = os.path.join(data_path,"val",cl,art + fname)
        else:
            new_dir = os.path.join(data_path,"train",cl,art + fname)
        count=count+1
        #print(d,new_dir)
        shutil.copy(d, new_dir)
    print("--------------")