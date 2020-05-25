from setuptools import find_packages, setup


def get_requirements():
    with open('requirements.txt') as fh:
        requirements = fh.read().splitlines()
    return requirements


setup(
    name='masked_transformer',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points='''
        [console_scripts]
        masked_transformer=masked_transformer.commands:main
    '''
)
