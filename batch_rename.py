# -*- coding: utf-8 -*-
import os
import sys


def rename():
    path = "D:\\sz4361\\公司项目文件\\舱门检测\\舱门视频\\舱门视频\\select\\three"
    name = "frame"
    fileType = ".jpg"
    count = 0
    filelist = os.listdir(path)
    for file in filelist:
        oldF = os.path.join(path, file)
        if os.path.isfile(oldF) and os.path.splitext(oldF)[1] == fileType:
            newF = os.path.join(
                path, "three" + file)
            os.rename(oldF, newF)
        else:
            continue
        count += 1
    print("一共修改了"+str(count)+"个文件")


rename()