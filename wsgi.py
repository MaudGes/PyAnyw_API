import sys
import os

# Add your project to the sys.path
project_home = '/home/MaudGes/mysite'
if project_home not in sys.path:
    sys.path.append(project_home)

# Add the virtualenv site-packages to sys.path
venv_site_packages = '/home/MaudGes/mysite/envp7/lib/python3.10/site-packages'
if venv_site_packages not in sys.path:
    sys.path.append(venv_site_packages)

# Import your Flask app
from app import app as application