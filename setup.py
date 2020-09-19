import setuptools

setuptools.setup(
    name='mlflow-pluggable-scoring-server',
    version='0.1',
    author='Andre M',
    description='MLflow pluggable scoring server',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'mlflow==1.11.0',
        'Flask==1.1.2',
        'flask-restplus==0.13.0'
    ]
)
