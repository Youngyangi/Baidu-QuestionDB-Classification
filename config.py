import os.path
import pathlib

pwd_path = pathlib.Path(os.path.abspath(__file__)).parent

source_dir = os.path.join(pwd_path, 'Data/百度题库')

output_dir = os.path.join(pwd_path, 'Data/Output')

history_source_dir = os.path.join(source_dir, '高中_历史/origin')
geo_source_dir = os.path.join(source_dir, '高中_地理/origin')
poli_source_dir = os.path.join(source_dir, '高中_政治/origin')
bio_source_dir = os.path.join(source_dir, '高中_生物/origin')

