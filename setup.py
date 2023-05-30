from setuptools import setup, find_packages


setup(name='Intra-camera similarity refinement',
      version='1.0.0',
      description='Pseudo labels refinement with intra-camera similarity for unsupervised person re-identification',
      author='Napeng Li',
      author_email='sauerfisch@stu.xjtu.edu.cn',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu',
          'pandas', 'matplotlib'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
      ])
