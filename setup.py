<<<<<<< HEAD
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT= "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements= []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n", " ") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
            name="Chat bot",
            version= "0.0.1",
            author= "Sonali Basak",
            description= "Chat gpt like clone",
            author_email="sonali62442@gmail.com",
            install_requires= get_requirements("requirements.txt"),
            packages= find_packages() 
=======
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT= "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements= []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n", " ") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
            name="Chat bot",
            version= "0.0.1",
            author= "Sonali Basak",
            description= "Chat gpt like clone",
            author_email="sonali62442@gmail.com",
            install_requires= get_requirements("requirements.txt"),
            packages= find_packages() 
>>>>>>> 11bb6b364243b5b0398b74a92c2da540688b395d
)