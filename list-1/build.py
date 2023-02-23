import os

if not os.path.exists('./images-out'):
    os.makedirs('./images-out')

from pip._internal import main as pipmain

pipmain(['install', '-r', 'requirements.txt'])