import sys 
try:
	from setuptools import setup
	have_setuptools = True 
except ImportError:
	from distutils.core import setup 
	have_setuptools = False

setup_kwargs = {
'name': 'CosMojo', 
'version': '0.1.0', 
'description': '', 
'author': 'Federico Bianchini', 
'author_email': 'federico.bianxini@gmail.com', 
'url': 'https://github.com/fbianchini/CosMojo', 
'packages':['cosmojo'],
'zip_safe': False, 
}

if __name__ == '__main__': 
	setup(**setup_kwargs)