#!/usr/bin/env python3
# coding:utf-8
from convertfilestomarkdown.utils.mammothutils import convert_to_html, img_element
from convertfilestomarkdown.utils import fileutils, htmlmdutils
import re
from tqdm import tqdm
import os
import mammoth
from convertfilestomarkdown.short_task_mp import short_task_mp, mp

@img_element
def data_uri(image, savepath, image_count):
    with image.open() as image_bytes:
        name = os.path.join(savepath, str(image_count))
        # print('%s.png'%name)
        with open('%s.png'%name, 'wb') as out:
            # 读取 ZIP 文件中的图片数据并写入新文件
            out.write(image_bytes.read())
    return {
        "src": '%s.png'%image_count
    }

def docx_to_md(doc_file, save_file):
    with open(doc_file, "rb") as docx_file:
        result = mammoth.convert_to_markdown(docx_file, convert_image=data_uri)
    fileutils.save_file(save_file, result.value)


def docx_to_html(doc_file, save_file):
    with open(doc_file, "rb") as docx_file:
        result = convert_to_html(docx_file, convert_image=data_uri)
    fileutils.save_file(save_file, result.value)

# def docx_to_md_ex(doc_file, save_file, dirname):
#     with open(doc_file, "rb") as docx_file:
        # result = mammoth.convert_to_html(docx_file)
#     html_str = result.value
#     md_str = htmlmdutils.html_to_text_ex(html_str)
#     fileutils.save_file(save_file, md_str)




# doc_file = r'd:\rag2025\知识库\50-计算-知识集\03-X板斧\【X板斧】昇腾计算互联网领域问题处理三板斧 V1.0-杨珊.docx'
# save_file = r'c:\Users\w30057236\Desktop\2\test.md'
def docx_to_md_ex(doc_file, save_file, dirname):
    print(save_file)
    # dirname = os.path.dirname(save_file)
    os.makedirs(dirname, exist_ok=True)
    try:
        with open(doc_file, "rb") as docx_file:
            result = convert_to_html(docx_file, convert_image=data_uri, save_file=dirname)
        html_str = result.value
        html_str = html_str.replace('#', '\#')      #替换掉#符号，防止markdown格式混乱
        md_str = htmlmdutils.html_to_text_ex(html_str)
        fileutils.save_file(save_file, md_str)
    except Exception as e:
        print(e)
# docx_to_md_ex(doc_file, save_file)

# doc_file = r'd:\rag2025\知识库\50-计算-知识集\03-X板斧\昇腾维护日志数据采集指导-2-NPU故障采集场景-V1.0.docx'
# save_file = r'c:\Users\w30057236\Desktop\1\test.md'
# dirname = r'c:\Users\w30057236\Desktop\1'
# from docx2md import convert
# def convert_docx_to_md(doc_file, save_file):
#     md_content = convert.do_convert(doc_file, os.path.dirname(save_file))
#     with open(save_file, 'w', encoding='utf-8') as f:
#         f.write(md_content)
# convert_docx_to_md(doc_file, save_file)
# 示例调用



def docx_to_md_with_info(data_folder=None, output_folder=None, product_name=None, doc_type="hdx"):
    save_file = fileutils.join(output_folder, product_name + ".md")
    docx_to_md_ex(doc_file=data_folder, save_file=save_file)

def docx_to_md_new(data_dir, target_root, input_paths=[], doc_type='docx', mp_pool=None, process_limit=0, coverage=False):
    # global data_file, target_md_path, dirname
    args_stack = []
    if not input_paths:
        input_paths = fileutils.get_target_paths(data_dir, [doc_type])
    for data_file in tqdm(input_paths):
        target_file = re.sub('\.%s$'%doc_type, '.md', data_file).replace(data_dir, target_root)
        dirname = os.path.dirname(target_file)
        filename = os.path.basename(target_file)
        dirname = os.path.join(dirname, re.sub('\.md$', '', filename))
        target_md_path = os.path.join(dirname, filename)
        if os.path.exists(target_md_path) and coverage==False:
            continue
        if mp_pool:
            if mp_pool == 'short_task':
                args_stack += [[data_file, target_md_path, dirname]]
            else:
                mp_pool.apply_async(docx_to_md_ex, args=(data_file, target_md_path, dirname))
        else:
            try:
                docx_to_md_ex(doc_file=data_file, save_file=target_md_path, dirname=dirname)
            except:
                pass
    if mp_pool == 'short_task':
        short_task_mp(docx_to_md_ex, args_stack, time_limit = 1000, process_limit=process_limit)

def check_empty_dir(target_root):
    listdir = os.listdir(target_root)
    for file in tqdm(listdir):
        filepath = os.path.join(target_root, file)
        sub_filelist = os.listdir(filepath)
        if len(sub_filelist)==0:
            os.rmdir(filepath)


if __name__=="__main__":
    
    # target_md_path = r'c:\Users\w30057236\Desktop\新建文件夹\无线绿色节能常见问题集_用户大会脱敏版.md'
    
    #------------------------docx转md-------------------------------
    # data_dir = r'D:\阅读伙伴_20240315\3gpp_word'
    data_dir = r'D:\--------阅读伙伴--------\增量预训练语料集'
    target_root = r'D:\--------阅读伙伴--------\md_files'

    # data_dir = '/data/markdown_seg/data/reading_partner_raw_data/3gpp_word'
    # target_root = '/data/markdown_seg/data/reading_partner/md_outputs_docx'
    
    process_limit = 12
    process_list = mp.Pool(processes=process_limit)
    
    docx_to_md_new(data_dir, target_root, 
                mp_pool='short_task', process_limit=process_limit)
    
    process_list.close()
    process_list.join()

    check_empty_dir(target_root)
    


