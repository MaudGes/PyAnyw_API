import sys
import os

# Add your project to the sys.path
project_home = '/home/home/MaudGes/mysite'
if project_home not in sys.path:
    sys.path.append(project_home)

# Activate your virtual environment
activate_this = '/home/home/MaudGes/mysite/envp7/bin/activate'
exec(open(activate_this).read(), dict(__file__=activate_this))

# Import your Flask app
from app import app as application
