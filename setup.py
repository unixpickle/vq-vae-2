from setuptools import setup

setup(
    name='vq-vae-2',
    version='0.0.1',
    description='A PyTorch implementation of VQ-VAE-2',
    url='https://github.com/unixpickle/vq-vae-2',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ai generative model',
    packages=['vq_vae_2'],
    install_requires=[
        'torch>=1.0.0',
        'torchvision>=0.2.1',
    ],
)
