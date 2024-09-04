import os
import shutil
import re

def remove_special_characters(filename):
    # 去除中文括弧、空格、逗号等特殊字符
    pattern = r'[^\w\d./-]'
    cleaned_filename = re.sub(pattern, '', filename)
    return cleaned_filename

root_dir = 'dataset_dir'
out_dir = 'dataset_dir_demucs/'

for _root, _dirs, _files in os.walk(root_dir):
    for _file in _files:
        path = os.path.join(_root, _file)
        new_path = remove_special_characters(path).replace('dataset_dir', 'dataset_dir_rename')
        new_dir = os.path.dirname(new_path)
        os.makedirs(new_dir, exist_ok=True)
        #os.system('cp %s %s' % (path, new_path))
        shutil.copy(path, new_path)
        #cmd = 'demucs %s -n mdx_extra --two-stems=vocals --out %s' % (new_path, out_dir)
        #cmd = 'demucs %s -n htdemucs_6s --two-stems=vocals --out %s' % (new_path, out_dir)
        cmd = 'demucs %s --two-stems=vocals -j 8 --out %s' % (new_path, out_dir)
        print(path)
        print(cmd)
        os.system(cmd)


