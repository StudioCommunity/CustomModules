import os
import os.path
import sys
from git import Repo 
import shutil
import tempfile

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Import Folder from git")
logger = logging.getLogger(__name__)
        

@click.command()
@click.option('--git_url')
@click.option('--out_path', default='out')
def run_pipeline(git_url, out_path):
    #temp_folder = tempfile.TemporaryDirectory()
    #temp_path = temp_folder.name
    temp_path = 'temp'
    Repo.clone_from(git_url, temp_path)
    if os.path.exists(out_path):
        for entry in os.scandir(temp_path):
            src = os.path.join(temp_path, entry.name)
            shutil.move(src, out_path)
    else:
        shutil.move(temp_path, out_path)
    #temp_folder.cleanup()
    print(f"OUTPUT({out_path}): {os.listdir(out_path)}")
    
# python -m dstest.importer.git2folder --git_url https://github.com/RichardZhaoW/Models.git --out_path out
if __name__ == '__main__':
    run_pipeline()