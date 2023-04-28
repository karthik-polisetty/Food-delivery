from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(filepath:str)->List[str]:
    requirements = []

    with open(filepath) as f:
        requirements = f.readlines()
        requirements= [req.replace("\n","") for req in requirements]

        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        

    
    return requirements
    




setup(
    name = 'Food-Delivery-Regression',
    version='0.0.1',
    author='karthik polisetty',
    author_email='kpolisetty1524@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
    )