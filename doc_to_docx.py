#!/usr/bin/env python3
# coding:utf-8
import os, shutil
from win32com.client import Dispatch, DispatchEx
from utils import fileutils
from tqdm import tqdm
from docx import Document
from utils.fileutils import get_target_paths

tmp_dir="tmp"
class WordProcess():

    def __init__(self):
        pass

    def start(self):
        # self.word = Dispatch("Word.Application")
        self.word = DispatchEx("Word.Application")

    def end(self):
        self.word.Quit()
        self.__delattr__('word')

    def convert(self,data_file,target_file):
        # 打开原始文档
        data_file="\"{}\"".format(data_file)
        data_file=data_file.replace(" ","\ ")
        doc = self.word.Documents.Open(data_file)
        target_extension=fileutils.file_extension(target_file)
        target_type=16
        doc.SaveAs(target_file, target_type)
        doc.Close()

    def convert_ex(self,data_file,target_file):
        tmp_file=fileutils.join(tmp_dir,"tmp.doc")
        tmp_rst=fileutils.join(tmp_dir,"tmp.docx")
        tmp_file=fileutils.join_with_folder(__file__,tmp_file)
        tmp_rst=fileutils.join_with_folder(__file__,tmp_rst)
        fileutils.ensure_parent(tmp_file)
        fileutils.copy_file(data_file,tmp_file)
        self.convert(tmp_file,tmp_rst)
        fileutils.copy_file(tmp_rst,target_file)





def doc_to_docx_ex(input, output):
    document=Document(input)
    document.save(output)
    print(output)

def doc_to_docx_(input, output, dustbin):
    wordprocess=WordProcess()
    wordprocess.start()
    print(input)
    try:
        wordprocess.convert(input,output)
        wordprocess.end()
    except Exception as e:
        print(e)
        shutil.copy(input, dustbin)
        # wordprocess.end()
    



if __name__=="__main__":
    import re
    path = r'C:\Users\w30057236\Desktop\testfiles'
    wordprocess=WordProcess()
    input_paths = get_target_paths(path, ['doc'])
    for data_file in tqdm(input_paths):
        # data_file=fileutils.join_with_folder(__file__,data_file)
        target_file = re.sub('\.doc$', '.docx', data_file)
        print(data_file)
        try:
            wordprocess.start()
            wordprocess.convert(data_file,target_file)
            wordprocess.end()
        except:
            pass
        try:
            os.remove(data_file)
        except:
            pass