import itertools
import json
import math
import pathlib
import traceback

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnx_tool
import onnxruntime

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

from plotly import tools

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/780d0b14.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a1570a43.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d90796e8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3f7978a0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/22eb0ac0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0520fde7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/23b5c85d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d9fac9be.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/890034e9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9ecd008a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/06df4c85.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2dc579da.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/007bbfb7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4522001f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/cdecee7f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/39e1d7f9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c3e719e8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b94a9452.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/846bdb03.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5c0a986e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/90c28cc7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/963e52fc.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f5b8619d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e9614598.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f25fbde4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7df24a62.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/469497ad.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/22233c11.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3befdf3e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6d0160f0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e8593010.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8be77c9e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/bb43febb.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/995c5fa3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/868de0fa.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4290ef0e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c3f564a4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c444b776.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/60b61512.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a8c38be5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/46f33fce.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8731374e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/80af3007.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/539a4f51.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d406998b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/22168020.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/00d62c1b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b548a754.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/913fb3ed.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/83302e8f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2013d3e2.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/28bf18c6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a79310a0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e9afcf9a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/54d82841.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/928ad970.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/eb281b96.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7b7f7511.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d631b094.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/444801d8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4258a5f9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6e02f1e3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a48eeaf7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d07ae81c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/50cb2852.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c59eb873.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/aba27056.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a65b410d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/11852cab.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/bdad9b1f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6d0aefbc.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5614dbcf.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a85d4709.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8e1813be.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7b6016b9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/77fdfe62.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a9f96cdd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a5f85a15.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8d510a79.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f35d900a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d2abd087.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/447fd412.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ea786f4a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/941d9a10.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a61ba2ce.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f25ffba3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b9b7f026.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/32597951.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b8825c91.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3bdb4ada.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a740d043.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7fe24cdd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d364b489.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/29623171.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3bd67248.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b782dc8a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/045e512c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c8f0f002.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1e32b0e9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1caeab9d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/be94b721.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ded97339.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1c786137.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ce4f8723.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7468f01a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8403a5d5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e6721834.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e50d258f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7f4411dc.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/95990924.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/264363fd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/746b3537.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4347f46a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/681b3aeb.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6c434453.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f1cefba8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9aec4887.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7c008303.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/de1cd16c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f8b3ba0a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ecdecbb3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3428a4f5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2bcee788.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/eb5a1d5d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1e0a9b12.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/31aa019c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b8cdaf2b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a3325580.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1b2d62fb.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1bfc4729.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4093f84a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6455b5f5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/29c11459.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/363442ee.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b2862040.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d4469b4b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d511f180.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e98196ab.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/662c240a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/445eab21.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b230c067.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1f642eb9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/794b24be.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ec883f72.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1b60fb0c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/56ff96f3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9edfc990.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/694f12f3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/73251a56.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6e19193c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ef135b50.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/36fdfd69.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/272f95fa.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d4a91cb9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/cbded52d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/bda2d7a6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/72ca375d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6fa7a44f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/137eaa0f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/97a05b5b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d13f3404.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d10ecb37.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9dfd6313.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8eb1be9a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6773b310.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c9f8e694.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ce602527.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/67e8384a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4612dd53.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a416b8f3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ba97ae07.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ce22a75a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4c5c2cf0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3af2c5a8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/dc0a314f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1f876c06.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/017c7c7b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4c4377d9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/48d8fb45.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6d75e8bb.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/239be575.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6ecd11f4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6d58a25d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/025d127b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8d5021e8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8efcae92.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3aa6fb7a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/234bbc79.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3c9b0459.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/68b16354.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2dee498d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2bee17df.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/67385a82.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d22278a0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/42a50994.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7e0986d6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a61f2674.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d89b689b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/28e73c20.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0a938d79.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e40b9e2f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a78176bb.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/44f52bb0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e5062a87.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/57aa92db.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/54d9e175.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5521c0d9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c1d99e64.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ea32f347.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1fad071e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/10fcaaa3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e3497940.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/91413438.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/484b58aa.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e8dc4411.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e179c5f4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/50846271.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3eda0437.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6855a6e4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/46442a0e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/27a28665.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/88a62173.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3ac3eb23.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c9e6f938.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7447852a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1f0c79e5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a68b268e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b27ca6d3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a3df8b1e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ac0a08a4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/834ec97d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/25d487eb.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6b9890af.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0ca9ddb6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c909285e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b775ac94.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5ad4f10b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/fafffa47.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2281f1f4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7837ac64.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/44d8ac46.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0dfd9992.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/150deff5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b7249182.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ff805c23.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5daaa586.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/90f3ed37.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6f8cd79b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/47c1f68c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6150a2bd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/dc1df850.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d4f3cd78.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/99b1bc43.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b527c5c6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/fcc82909.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6a1e5592.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/25ff71a9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c8cbb738.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a2fd1cf0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ae4f1146.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b190f7f5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d8c310e9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/caa06a1f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f8a8fe49.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5168d44c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/cf98881b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6430c8c4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9565186b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/56dc2b01.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/25d8a9c8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a64e4611.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ae3edfdc.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3631a71a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/253bf280.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a87f7484.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/321b1fc6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9af7a82c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d687bc17.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e26a3af2.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/75b8110e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/72322fa7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/dc433765.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2dd70a9a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/673ef223.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/7ddcd7ec.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/82819916.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/beb8660c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6aa20dc0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5bd6f4ac.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d037b0a7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/08ed6ac7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/cce03e0d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/aedd82e4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3e980e27.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8a004b2b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f8ff0b80.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b0c4d837.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/09629e4f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/855e0971.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/228f6490.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/88a10436.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/67a423a3.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9f236235.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/810b9b61.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5c2c9af4.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0e206a2e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/af902bf9.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f76d97a5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/fcb5c309.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/aabf363d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/63613498.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e48d4e1a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/178fcbfb.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1f85a75f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ff28f65a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/508bd3b6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a699fb00.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/29ec7d0e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e76a88a6.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9d9215db.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ed36ccf7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/bbc9ae5d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/db3e9e38.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d9f24cd1.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/623ea044.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f15e1fac.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1190e5a7.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/49d1d64f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e509e548.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d43fd935.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/05269061.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e21d9049.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a8d7556c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/85c4e7cd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5117e062.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/776ffc46.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/5582e5ca.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2204b7a8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/2c608aff.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1cf80156.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/05f2a901.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/760b3cac.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/93b581b8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6cf79266.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3de23699.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/dae9d2b5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/a5313dff.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d23f8c26.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/db93a21d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f9012d9b.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/39a8645d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/67a3c6ac.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4938f0c2.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/9172f3a0.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0962bcdd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/40853293.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ce9e57f2.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d06dbe63.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/feca6190.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f2829549.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ddf7fa4f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/91714a58.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/bd4472b8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/62c24649.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3345333e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/dbc1a6ce.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6e82a1ae.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/952a094c.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b1948b0a.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8f2ea7aa.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/6cdd2623.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0d3d703e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b6afb2da.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/23581191.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/c0f76784.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/543a7ed5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/99fa7670.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/41e4d17e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d5d6de2d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/8e5a5113.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/e73095fd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/bc1d5164.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/97999447.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d0f5fe59.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b60334d2.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/53b68214.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/4be741c5.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/1a07d186.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3618c87e.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/f8c80d96.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/3906de3d.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/94f9d214.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/496994bd.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/74dd1130.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/98cf29f8.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/d6ad076f.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/b91ae062.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/0b148d64.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/36d67576.json
/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/ba26e723.json
/kaggle/input/competitions/neurogolf-2026/task221.json
/kaggle/input/competitions/neurogolf-2026/task189.json
/kaggle/input/competitions/neurogolf-2026/task292.json
/kaggle/input/competitions/neurogolf-2026/task176.json
/kaggle/input/competitions/neurogolf-2026/task210.json
/kaggle/input/competitions/neurogolf-2026/task363.json
/kaggle/input/competitions/neurogolf-2026/task179.json
/kaggle/input/competitions/neurogolf-2026/task154.json
/kaggle/input/competitions/neurogolf-2026/task357.json
/kaggle/input/competitions/neurogolf-2026/task304.json
/kaggle/input/competitions/neurogolf-2026/task022.json
/kaggle/input/competitions/neurogolf-2026/task090.json
/kaggle/input/competitions/neurogolf-2026/task115.json
/kaggle/input/competitions/neurogolf-2026/task076.json
/kaggle/input/competitions/neurogolf-2026/task329.json
/kaggle/input/competitions/neurogolf-2026/task224.json
/kaggle/input/competitions/neurogolf-2026/task166.json
/kaggle/input/competitions/neurogolf-2026/task169.json
/kaggle/input/competitions/neurogolf-2026/task270.json
/kaggle/input/competitions/neurogolf-2026/task041.json
/kaggle/input/competitions/neurogolf-2026/task242.json
/kaggle/input/competitions/neurogolf-2026/task318.json
/kaggle/input/competitions/neurogolf-2026/task014.json
/kaggle/input/competitions/neurogolf-2026/task230.json
/kaggle/input/competitions/neurogolf-2026/task277.json
/kaggle/input/competitions/neurogolf-2026/task214.json
/kaggle/input/competitions/neurogolf-2026/task197.json
/kaggle/input/competitions/neurogolf-2026/task106.json
/kaggle/input/competitions/neurogolf-2026/task300.json
/kaggle/input/competitions/neurogolf-2026/task150.json
/kaggle/input/competitions/neurogolf-2026/task382.json
/kaggle/input/competitions/neurogolf-2026/task199.json
/kaggle/input/competitions/neurogolf-2026/task111.json
/kaggle/input/competitions/neurogolf-2026/task108.json
/kaggle/input/competitions/neurogolf-2026/task102.json
/kaggle/input/competitions/neurogolf-2026/task274.json
/kaggle/input/competitions/neurogolf-2026/task282.json
/kaggle/input/competitions/neurogolf-2026/task087.json
/kaggle/input/competitions/neurogolf-2026/task383.json
/kaggle/input/competitions/neurogolf-2026/task109.json
/kaggle/input/competitions/neurogolf-2026/task121.json
/kaggle/input/competitions/neurogolf-2026/task127.json
/kaggle/input/competitions/neurogolf-2026/task139.json
/kaggle/input/competitions/neurogolf-2026/task202.json
/kaggle/input/competitions/neurogolf-2026/task017.json
/kaggle/input/competitions/neurogolf-2026/task379.json
/kaggle/input/competitions/neurogolf-2026/task074.json
/kaggle/input/competitions/neurogolf-2026/task081.json
/kaggle/input/competitions/neurogolf-2026/task117.json
/kaggle/input/competitions/neurogolf-2026/task368.json
/kaggle/input/competitions/neurogolf-2026/task059.json
/kaggle/input/competitions/neurogolf-2026/task158.json
/kaggle/input/competitions/neurogolf-2026/task002.json
/kaggle/input/competitions/neurogolf-2026/task248.json
/kaggle/input/competitions/neurogolf-2026/task237.json
/kaggle/input/competitions/neurogolf-2026/task222.json
/kaggle/input/competitions/neurogolf-2026/task066.json
/kaggle/input/competitions/neurogolf-2026/task078.json
/kaggle/input/competitions/neurogolf-2026/task119.json
/kaggle/input/competitions/neurogolf-2026/task201.json
/kaggle/input/competitions/neurogolf-2026/task198.json
/kaggle/input/competitions/neurogolf-2026/task257.json
/kaggle/input/competitions/neurogolf-2026/task243.json
/kaggle/input/competitions/neurogolf-2026/task395.json
/kaggle/input/competitions/neurogolf-2026/task309.json
/kaggle/input/competitions/neurogolf-2026/task145.json
/kaggle/input/competitions/neurogolf-2026/task100.json
/kaggle/input/competitions/neurogolf-2026/task027.json
/kaggle/input/competitions/neurogolf-2026/task203.json
/kaggle/input/competitions/neurogolf-2026/task162.json
/kaggle/input/competitions/neurogolf-2026/task264.json
/kaggle/input/competitions/neurogolf-2026/task082.json
/kaggle/input/competitions/neurogolf-2026/task399.json
/kaggle/input/competitions/neurogolf-2026/task018.json
/kaggle/input/competitions/neurogolf-2026/task200.json
/kaggle/input/competitions/neurogolf-2026/task020.json
/kaggle/input/competitions/neurogolf-2026/task032.json
/kaggle/input/competitions/neurogolf-2026/task103.json
/kaggle/input/competitions/neurogolf-2026/task209.json
/kaggle/input/competitions/neurogolf-2026/task181.json
/kaggle/input/competitions/neurogolf-2026/task070.json
/kaggle/input/competitions/neurogolf-2026/task223.json
/kaggle/input/competitions/neurogolf-2026/task289.json
/kaggle/input/competitions/neurogolf-2026/task335.json
/kaggle/input/competitions/neurogolf-2026/task093.json
/kaggle/input/competitions/neurogolf-2026/task320.json
/kaggle/input/competitions/neurogolf-2026/task047.json
/kaggle/input/competitions/neurogolf-2026/task168.json
/kaggle/input/competitions/neurogolf-2026/task028.json
/kaggle/input/competitions/neurogolf-2026/task396.json
/kaggle/input/competitions/neurogolf-2026/task116.json
/kaggle/input/competitions/neurogolf-2026/task072.json
/kaggle/input/competitions/neurogolf-2026/task344.json
/kaggle/input/competitions/neurogolf-2026/task321.json
/kaggle/input/competitions/neurogolf-2026/task388.json
/kaggle/input/competitions/neurogolf-2026/task191.json
/kaggle/input/competitions/neurogolf-2026/task240.json
/kaggle/input/competitions/neurogolf-2026/task347.json
/kaggle/input/competitions/neurogolf-2026/task204.json
/kaggle/input/competitions/neurogolf-2026/task372.json
/kaggle/input/competitions/neurogolf-2026/task094.json
/kaggle/input/competitions/neurogolf-2026/task155.json
/kaggle/input/competitions/neurogolf-2026/task048.json
/kaggle/input/competitions/neurogolf-2026/task061.json
/kaggle/input/competitions/neurogolf-2026/task384.json
/kaggle/input/competitions/neurogolf-2026/task212.json
/kaggle/input/competitions/neurogolf-2026/task255.json
/kaggle/input/competitions/neurogolf-2026/task213.json
/kaggle/input/competitions/neurogolf-2026/task325.json
/kaggle/input/competitions/neurogolf-2026/task281.json
/kaggle/input/competitions/neurogolf-2026/task046.json
/kaggle/input/competitions/neurogolf-2026/task239.json
/kaggle/input/competitions/neurogolf-2026/task385.json
/kaggle/input/competitions/neurogolf-2026/task036.json
/kaggle/input/competitions/neurogolf-2026/task267.json
/kaggle/input/competitions/neurogolf-2026/task360.json
/kaggle/input/competitions/neurogolf-2026/task301.json
/kaggle/input/competitions/neurogolf-2026/task195.json
/kaggle/input/competitions/neurogolf-2026/task142.json
/kaggle/input/competitions/neurogolf-2026/task314.json
/kaggle/input/competitions/neurogolf-2026/task244.json
/kaggle/input/competitions/neurogolf-2026/task148.json
/kaggle/input/competitions/neurogolf-2026/task228.json
/kaggle/input/competitions/neurogolf-2026/task141.json
/kaggle/input/competitions/neurogolf-2026/task256.json
/kaggle/input/competitions/neurogolf-2026/task367.json
/kaggle/input/competitions/neurogolf-2026/task010.json
/kaggle/input/competitions/neurogolf-2026/task374.json
/kaggle/input/competitions/neurogolf-2026/task170.json
/kaggle/input/competitions/neurogolf-2026/task206.json
/kaggle/input/competitions/neurogolf-2026/task033.json
/kaggle/input/competitions/neurogolf-2026/task226.json
/kaggle/input/competitions/neurogolf-2026/task371.json
/kaggle/input/competitions/neurogolf-2026/task092.json
/kaggle/input/competitions/neurogolf-2026/task353.json
/kaggle/input/competitions/neurogolf-2026/task026.json
/kaggle/input/competitions/neurogolf-2026/task265.json
/kaggle/input/competitions/neurogolf-2026/task316.json
/kaggle/input/competitions/neurogolf-2026/task334.json
/kaggle/input/competitions/neurogolf-2026/task284.json
/kaggle/input/competitions/neurogolf-2026/task008.json
/kaggle/input/competitions/neurogolf-2026/task140.json
/kaggle/input/competitions/neurogolf-2026/task336.json
/kaggle/input/competitions/neurogolf-2026/task077.json
/kaggle/input/competitions/neurogolf-2026/task086.json
/kaggle/input/competitions/neurogolf-2026/task143.json
/kaggle/input/competitions/neurogolf-2026/task346.json
/kaggle/input/competitions/neurogolf-2026/task291.json
/kaggle/input/competitions/neurogolf-2026/task030.json
/kaggle/input/competitions/neurogolf-2026/task164.json
/kaggle/input/competitions/neurogolf-2026/task185.json
/kaggle/input/competitions/neurogolf-2026/task062.json
/kaggle/input/competitions/neurogolf-2026/task269.json
/kaggle/input/competitions/neurogolf-2026/task144.json
/kaggle/input/competitions/neurogolf-2026/task174.json
/kaggle/input/competitions/neurogolf-2026/task295.json
/kaggle/input/competitions/neurogolf-2026/task069.json
/kaggle/input/competitions/neurogolf-2026/task134.json
/kaggle/input/competitions/neurogolf-2026/task096.json
/kaggle/input/competitions/neurogolf-2026/task104.json
/kaggle/input/competitions/neurogolf-2026/task160.json
/kaggle/input/competitions/neurogolf-2026/task225.json
/kaggle/input/competitions/neurogolf-2026/task006.json
/kaggle/input/competitions/neurogolf-2026/task112.json
/kaggle/input/competitions/neurogolf-2026/task211.json
/kaggle/input/competitions/neurogolf-2026/task251.json
/kaggle/input/competitions/neurogolf-2026/task352.json
/kaggle/input/competitions/neurogolf-2026/task358.json
/kaggle/input/competitions/neurogolf-2026/task038.json
/kaggle/input/competitions/neurogolf-2026/task341.json
/kaggle/input/competitions/neurogolf-2026/task254.json
/kaggle/input/competitions/neurogolf-2026/task180.json
/kaggle/input/competitions/neurogolf-2026/task327.json
/kaggle/input/competitions/neurogolf-2026/task249.json
/kaggle/input/competitions/neurogolf-2026/task123.json
/kaggle/input/competitions/neurogolf-2026/task159.json
/kaggle/input/competitions/neurogolf-2026/task311.json
/kaggle/input/competitions/neurogolf-2026/task280.json
/kaggle/input/competitions/neurogolf-2026/task193.json
/kaggle/input/competitions/neurogolf-2026/task247.json
/kaggle/input/competitions/neurogolf-2026/task216.json
/kaggle/input/competitions/neurogolf-2026/task323.json
/kaggle/input/competitions/neurogolf-2026/task126.json
/kaggle/input/competitions/neurogolf-2026/task343.json
/kaggle/input/competitions/neurogolf-2026/task025.json
/kaggle/input/competitions/neurogolf-2026/task279.json
/kaggle/input/competitions/neurogolf-2026/task040.json
/kaggle/input/competitions/neurogolf-2026/task380.json
/kaggle/input/competitions/neurogolf-2026/task303.json
/kaggle/input/competitions/neurogolf-2026/task362.json
/kaggle/input/competitions/neurogolf-2026/task095.json
/kaggle/input/competitions/neurogolf-2026/task337.json
/kaggle/input/competitions/neurogolf-2026/task393.json
/kaggle/input/competitions/neurogolf-2026/task165.json
/kaggle/input/competitions/neurogolf-2026/task177.json
/kaggle/input/competitions/neurogolf-2026/task286.json
/kaggle/input/competitions/neurogolf-2026/task276.json
/kaggle/input/competitions/neurogolf-2026/task313.json
/kaggle/input/competitions/neurogolf-2026/task293.json
/kaggle/input/competitions/neurogolf-2026/task192.json
/kaggle/input/competitions/neurogolf-2026/task273.json
/kaggle/input/competitions/neurogolf-2026/task250.json
/kaggle/input/competitions/neurogolf-2026/task315.json
/kaggle/input/competitions/neurogolf-2026/task019.json
/kaggle/input/competitions/neurogolf-2026/task122.json
/kaggle/input/competitions/neurogolf-2026/task110.json
/kaggle/input/competitions/neurogolf-2026/task023.json
/kaggle/input/competitions/neurogolf-2026/task063.json
/kaggle/input/competitions/neurogolf-2026/task263.json
/kaggle/input/competitions/neurogolf-2026/task351.json
/kaggle/input/competitions/neurogolf-2026/task296.json
/kaggle/input/competitions/neurogolf-2026/task058.json
/kaggle/input/competitions/neurogolf-2026/task132.json
/kaggle/input/competitions/neurogolf-2026/task161.json
/kaggle/input/competitions/neurogolf-2026/task029.json
/kaggle/input/competitions/neurogolf-2026/task044.json
/kaggle/input/competitions/neurogolf-2026/task011.json
/kaggle/input/competitions/neurogolf-2026/task064.json
/kaggle/input/competitions/neurogolf-2026/task073.json
/kaggle/input/competitions/neurogolf-2026/task287.json
/kaggle/input/competitions/neurogolf-2026/task186.json
/kaggle/input/competitions/neurogolf-2026/task220.json
/kaggle/input/competitions/neurogolf-2026/task258.json
/kaggle/input/competitions/neurogolf-2026/task369.json
/kaggle/input/competitions/neurogolf-2026/task391.json
/kaggle/input/competitions/neurogolf-2026/task361.json
/kaggle/input/competitions/neurogolf-2026/task075.json
/kaggle/input/competitions/neurogolf-2026/task153.json
/kaggle/input/competitions/neurogolf-2026/task299.json
/kaggle/input/competitions/neurogolf-2026/task235.json
/kaggle/input/competitions/neurogolf-2026/task188.json
/kaggle/input/competitions/neurogolf-2026/task394.json
/kaggle/input/competitions/neurogolf-2026/task268.json
/kaggle/input/competitions/neurogolf-2026/task101.json
/kaggle/input/competitions/neurogolf-2026/task261.json
/kaggle/input/competitions/neurogolf-2026/task217.json
/kaggle/input/competitions/neurogolf-2026/task053.json
/kaggle/input/competitions/neurogolf-2026/task266.json
/kaggle/input/competitions/neurogolf-2026/task386.json
/kaggle/input/competitions/neurogolf-2026/task015.json
/kaggle/input/competitions/neurogolf-2026/task107.json
/kaggle/input/competitions/neurogolf-2026/task157.json
/kaggle/input/competitions/neurogolf-2026/task317.json
/kaggle/input/competitions/neurogolf-2026/task114.json
/kaggle/input/competitions/neurogolf-2026/task322.json
/kaggle/input/competitions/neurogolf-2026/task013.json
/kaggle/input/competitions/neurogolf-2026/task356.json
/kaggle/input/competitions/neurogolf-2026/task229.json
/kaggle/input/competitions/neurogolf-2026/task227.json
/kaggle/input/competitions/neurogolf-2026/task068.json
/kaggle/input/competitions/neurogolf-2026/task178.json
/kaggle/input/competitions/neurogolf-2026/task234.json
/kaggle/input/competitions/neurogolf-2026/task308.json
/kaggle/input/competitions/neurogolf-2026/task056.json
/kaggle/input/competitions/neurogolf-2026/task348.json
/kaggle/input/competitions/neurogolf-2026/task378.json
/kaggle/input/competitions/neurogolf-2026/task105.json
/kaggle/input/competitions/neurogolf-2026/task021.json
/kaggle/input/competitions/neurogolf-2026/task163.json
/kaggle/input/competitions/neurogolf-2026/task326.json
/kaggle/input/competitions/neurogolf-2026/task089.json
/kaggle/input/competitions/neurogolf-2026/task067.json
/kaggle/input/competitions/neurogolf-2026/task236.json
/kaggle/input/competitions/neurogolf-2026/task294.json
/kaggle/input/competitions/neurogolf-2026/task328.json
/kaggle/input/competitions/neurogolf-2026/task389.json
/kaggle/input/competitions/neurogolf-2026/task113.json
/kaggle/input/competitions/neurogolf-2026/task175.json
/kaggle/input/competitions/neurogolf-2026/task205.json
/kaggle/input/competitions/neurogolf-2026/task298.json
/kaggle/input/competitions/neurogolf-2026/task365.json
/kaggle/input/competitions/neurogolf-2026/task208.json
/kaggle/input/competitions/neurogolf-2026/task207.json
/kaggle/input/competitions/neurogolf-2026/task246.json
/kaggle/input/competitions/neurogolf-2026/task373.json
/kaggle/input/competitions/neurogolf-2026/task016.json
/kaggle/input/competitions/neurogolf-2026/task400.json
/kaggle/input/competitions/neurogolf-2026/task183.json
/kaggle/input/competitions/neurogolf-2026/task392.json
/kaggle/input/competitions/neurogolf-2026/task131.json
/kaggle/input/competitions/neurogolf-2026/task042.json
/kaggle/input/competitions/neurogolf-2026/task387.json
/kaggle/input/competitions/neurogolf-2026/task054.json
/kaggle/input/competitions/neurogolf-2026/task307.json
/kaggle/input/competitions/neurogolf-2026/task345.json
/kaggle/input/competitions/neurogolf-2026/task332.json
/kaggle/input/competitions/neurogolf-2026/task128.json
/kaggle/input/competitions/neurogolf-2026/task272.json
/kaggle/input/competitions/neurogolf-2026/task051.json
/kaggle/input/competitions/neurogolf-2026/task370.json
/kaggle/input/competitions/neurogolf-2026/task173.json
/kaggle/input/competitions/neurogolf-2026/task375.json
/kaggle/input/competitions/neurogolf-2026/task397.json
/kaggle/input/competitions/neurogolf-2026/task037.json
/kaggle/input/competitions/neurogolf-2026/task043.json
/kaggle/input/competitions/neurogolf-2026/task194.json
/kaggle/input/competitions/neurogolf-2026/task079.json
/kaggle/input/competitions/neurogolf-2026/task381.json
/kaggle/input/competitions/neurogolf-2026/task085.json
/kaggle/input/competitions/neurogolf-2026/task285.json
/kaggle/input/competitions/neurogolf-2026/task012.json
/kaggle/input/competitions/neurogolf-2026/task133.json
/kaggle/input/competitions/neurogolf-2026/task049.json
/kaggle/input/competitions/neurogolf-2026/task350.json
/kaggle/input/competitions/neurogolf-2026/task290.json
/kaggle/input/competitions/neurogolf-2026/task065.json
/kaggle/input/competitions/neurogolf-2026/task245.json
/kaggle/input/competitions/neurogolf-2026/task099.json
/kaggle/input/competitions/neurogolf-2026/task283.json
/kaggle/input/competitions/neurogolf-2026/task278.json
/kaggle/input/competitions/neurogolf-2026/task262.json
/kaggle/input/competitions/neurogolf-2026/task271.json
/kaggle/input/competitions/neurogolf-2026/task152.json
/kaggle/input/competitions/neurogolf-2026/task259.json
/kaggle/input/competitions/neurogolf-2026/task156.json
/kaggle/input/competitions/neurogolf-2026/task045.json
/kaggle/input/competitions/neurogolf-2026/task333.json
/kaggle/input/competitions/neurogolf-2026/task135.json
/kaggle/input/competitions/neurogolf-2026/task031.json
/kaggle/input/competitions/neurogolf-2026/task190.json
/kaggle/input/competitions/neurogolf-2026/task124.json
/kaggle/input/competitions/neurogolf-2026/task151.json
/kaggle/input/competitions/neurogolf-2026/task349.json
/kaggle/input/competitions/neurogolf-2026/task167.json
/kaggle/input/competitions/neurogolf-2026/task231.json
/kaggle/input/competitions/neurogolf-2026/task138.json
/kaggle/input/competitions/neurogolf-2026/task172.json
/kaggle/input/competitions/neurogolf-2026/task364.json
/kaggle/input/competitions/neurogolf-2026/task050.json
/kaggle/input/competitions/neurogolf-2026/task003.json
/kaggle/input/competitions/neurogolf-2026/task060.json
/kaggle/input/competitions/neurogolf-2026/task339.json
/kaggle/input/competitions/neurogolf-2026/task184.json
/kaggle/input/competitions/neurogolf-2026/task302.json
/kaggle/input/competitions/neurogolf-2026/task080.json
/kaggle/input/competitions/neurogolf-2026/task052.json
/kaggle/input/competitions/neurogolf-2026/task055.json
/kaggle/input/competitions/neurogolf-2026/task338.json
/kaggle/input/competitions/neurogolf-2026/task376.json
/kaggle/input/competitions/neurogolf-2026/task088.json
/kaggle/input/competitions/neurogolf-2026/task390.json
/kaggle/input/competitions/neurogolf-2026/task118.json
/kaggle/input/competitions/neurogolf-2026/task136.json
/kaggle/input/competitions/neurogolf-2026/task233.json
/kaggle/input/competitions/neurogolf-2026/task129.json
/kaggle/input/competitions/neurogolf-2026/task377.json
/kaggle/input/competitions/neurogolf-2026/task149.json
/kaggle/input/competitions/neurogolf-2026/task355.json
/kaggle/input/competitions/neurogolf-2026/task342.json
/kaggle/input/competitions/neurogolf-2026/task009.json
/kaggle/input/competitions/neurogolf-2026/task024.json
/kaggle/input/competitions/neurogolf-2026/task241.json
/kaggle/input/competitions/neurogolf-2026/task083.json
/kaggle/input/competitions/neurogolf-2026/task330.json
/kaggle/input/competitions/neurogolf-2026/task007.json
/kaggle/input/competitions/neurogolf-2026/task039.json
/kaggle/input/competitions/neurogolf-2026/task001.json
/kaggle/input/competitions/neurogolf-2026/task137.json
/kaggle/input/competitions/neurogolf-2026/task035.json
/kaggle/input/competitions/neurogolf-2026/task171.json
/kaggle/input/competitions/neurogolf-2026/task219.json
/kaggle/input/competitions/neurogolf-2026/task057.json
/kaggle/input/competitions/neurogolf-2026/task359.json
/kaggle/input/competitions/neurogolf-2026/task187.json
/kaggle/input/competitions/neurogolf-2026/task005.json
/kaggle/input/competitions/neurogolf-2026/task331.json
/kaggle/input/competitions/neurogolf-2026/task004.json
/kaggle/input/competitions/neurogolf-2026/task354.json
/kaggle/input/competitions/neurogolf-2026/task253.json
/kaggle/input/competitions/neurogolf-2026/task312.json
/kaggle/input/competitions/neurogolf-2026/task275.json
/kaggle/input/competitions/neurogolf-2026/task288.json
/kaggle/input/competitions/neurogolf-2026/task319.json
/kaggle/input/competitions/neurogolf-2026/task130.json
/kaggle/input/competitions/neurogolf-2026/task182.json
/kaggle/input/competitions/neurogolf-2026/task366.json
/kaggle/input/competitions/neurogolf-2026/task324.json
/kaggle/input/competitions/neurogolf-2026/task306.json
/kaggle/input/competitions/neurogolf-2026/task084.json
/kaggle/input/competitions/neurogolf-2026/task034.json
/kaggle/input/competitions/neurogolf-2026/task196.json
/kaggle/input/competitions/neurogolf-2026/task218.json
/kaggle/input/competitions/neurogolf-2026/task098.json
/kaggle/input/competitions/neurogolf-2026/task147.json
/kaggle/input/competitions/neurogolf-2026/task146.json
/kaggle/input/competitions/neurogolf-2026/task340.json
/kaggle/input/competitions/neurogolf-2026/task252.json
/kaggle/input/competitions/neurogolf-2026/task091.json
/kaggle/input/competitions/neurogolf-2026/task398.json
/kaggle/input/competitions/neurogolf-2026/task260.json
/kaggle/input/competitions/neurogolf-2026/task238.json
/kaggle/input/competitions/neurogolf-2026/task125.json
/kaggle/input/competitions/neurogolf-2026/task232.json
/kaggle/input/competitions/neurogolf-2026/task305.json
/kaggle/input/competitions/neurogolf-2026/task297.json
/kaggle/input/competitions/neurogolf-2026/task310.json
/kaggle/input/competitions/neurogolf-2026/task071.json
/kaggle/input/competitions/neurogolf-2026/task215.json
/kaggle/input/competitions/neurogolf-2026/task120.json
/kaggle/input/competitions/neurogolf-2026/task097.json
/kaggle/input/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py
ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus
Author: Michael D. Moffitt

"The Abstraction and Reasoning Corpus remains one of the most compelling and challenging benchmarks for tracking progress toward achieving Artificial General Intelligence. In contrast to other evaluation datasets designed to assess an agent's task-specific skills or accumulated knowledge, the ARC-AGI suite is specifically targeted at measuring skill acquisition efficiency, a trait that has (so far) been lacking in even the most sophisticated machine learning systems."

"For algorithms that require extensive intra-task exemplars, a significant constraint imposed by ARC-AGI is the modest cardinality of its demonstration set, comprising a small number of input, output grids per task specifying the corresponding transformation. To embellish the space of viable sample pairs, this paper introduces ARC-GEN, an open-source procedural generator aimed at extending the original ARC-AGI training dataset as faithfully as possible. Unlike prior efforts, the author generator is both exhaustive (covering all four-hundred tasks) and mimetic (more closely honoring the distributional properties and characteristics embodied in the initial ARC-AGI-1 release)."

"Each sub-generator included in their open-source release is not only capable of producing new mimetic examples, but also parameterized in such a way that allows the complete reproduction of the original benchmark suite, as well as the creation of a broader variety oftasks. Their library has many potential uses, including (but not limited to) the verification of programs designed to generalize beyond the specific examples included in the original ARC-AGI-1 distribution."

"The authors looked forward to seeing the use of their tool, along with others, in the pursuit of systems devoted to tackling Artificial General Intelligence."

"They also discussed the use of this generator in establishing a static benchmark suite to verify the correctness of programs submitted to the 2025 Google Code Golf Championship."

https://arxiv.org/abs/2511.00162

Competition Citation
@misc{neurogolf-2026,

author = {Michael D. Moffitt and Walter Reade and Ashley Oldacre and Addison Howard},
title = {The 2026 NeuroGolf Championship},

year = {2026},

howpublished = {\url{https://kaggle.com/competitions/neurogolf-2026}},
note = {Kaggle}
}

Since I ruined ARC-AGI, ONNX and Golf ARC 2025, let's get AI help


#JSON
import json

import polars  as pl
The 2026 NeuroGolf Championship
Design the smallest neural networks to solve ARC-AGI image transformations

"Solving a task is only the first step. Doing it efficiently is harder.

https://www.kaggle.com/competitions/neurogolf-2026/overview

"The objective of this competition is to create a suite of neural networks to implement a variety of transformations, where each transformation is implicitly described by a series of image grids. For example, the example pairs for one task might demonstrate the concept of rotation, whereas another might involve cropping and/or magnification. Your network for a given task should not only achieve the desired result across all exemplars, but also do so using the simplest possible architecture."

https://www.kaggle.com/competitions/neurogolf-2026/data

Task files
"The information for each of the four-hundred tasks is stored in an appropriately named json file (e.g., task001.json, task002.json). The file for a given task contains a dictionary with three fields:"

"train": a list of input/output pairs originally included in ARC-AGI-1 for training

"test": a list of input/output pairs originally included in ARC-AGI-1 for testing

"arc-gen": a list of additional input/output pairs included in The ARC-GEN-100K dataset

Open one json task file.
df = pd.read_json('/kaggle/input/competitions/neurogolf-2026/task001.json', typ="series")

df.tail()
train      [{'input': [[0, 7, 7], [7, 7, 7], [0, 7, 7]], ...
test       [{'input': [[7, 0, 7], [7, 0, 7], [7, 7, 0]], ...
arc-gen    [{'input': [[7, 7, 7], [0, 7, 0], [0, 7, 7]], ...
dtype: object
The ARC GEN 100K dataset
A set of 100,000 example pairs (covering all four hundred ARC-AGI-1 training tasks) produced by ARC-GEN.

https://www.kaggle.com/datasets/arcgen100k/the-arc-gen-100k-dataset

Open One json file from The ARC-GEN-100k Dataset
arc = pd.read_json('/kaggle/input/datasets/arcgen100k/the-arc-gen-100k-dataset/007bbfb7.json', typ="series")

arc.tail()
257    {'input': [[0, 0, 4], [0, 4, 0], [0, 4, 0]], '...
258    {'input': [[3, 3, 3], [0, 3, 0], [3, 0, 0]], '...
259    {'input': [[0, 9, 0], [9, 0, 0], [9, 9, 0]], '...
260    {'input': [[2, 0, 0], [0, 2, 0], [0, 0, 0]], '...
261    {'input': [[0, 0, 0], [0, 8, 0], [8, 0, 8]], '...
dtype: object
Install ONNXruntime and ONNX-tool
!pip install onnxruntime
!pip install onnx-tool
Import Libraries
import itertools
import json
import math
import pathlib
import traceback

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnx_tool
import onnxruntime
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

display = IPython.display.display
FileLink = IPython.display.FileLink

_BATCH_SIZE, _CHANNELS, _HEIGHT, _WIDTH = 1, 10, 30, 30
_NEUROGOLF_DIR = "/kaggle/input/competitions/neurogolf-2026/"
_COLORS = [
    (0, 0, 0),
    (30, 147, 255),
    (250, 61, 49),
    (78, 204, 48),
    (255, 221, 0),
    (153, 153, 153),
    (229, 59, 163),
    (255, 133, 28),
    (136, 216, 241),
    (147, 17, 49),
    (240, 240, 240),
    (146, 117, 86)
]
_DATA_TYPE = onnx.TensorProto.FLOAT
_EXCLUDED_OP_TYPES = ["LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION"]
_FILESIZE_LIMIT_IN_BYTES = 1.44 * 1024 * 1024
_GRID_SHAPE = [_BATCH_SIZE, _CHANNELS, _HEIGHT, _WIDTH]
_IR_VERSION, _OPSET_IMPORTS = 10, [onnx.helper.make_opsetid("", 10)]
_TASK_ZERO = {
    "train": [{
        "input": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        "output": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 5, 0, 0, 0, 0, 0, 0, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
    }],
    "test": [{
        "input": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 5, 5, 5],
            [5, 5, 4, 5, 5, 5, 4, 5, 5, 5],
            [5, 5, 4, 5, 5, 5, 4, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        "output": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 0, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 5, 5, 5],
            [5, 5, 4, 0, 0, 0, 4, 0, 5, 5],
            [5, 5, 4, 0, 5, 5, 4, 0, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 0, 5, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 5, 5],
        ],
    }],
    "arc-gen": [{
        "input": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        "output": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 5, 5],
            [5, 5, 2, 0, 0, 0, 0, 2, 0, 5],
            [5, 5, 2, 0, 5, 5, 5, 2, 0, 5],
            [5, 5, 2, 0, 5, 5, 5, 2, 0, 5],
            [5, 5, 2, 0, 5, 5, 5, 2, 0, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 0, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
    }],
}
Helper Functions that I don't know HOW they Function : (
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def check_network(filename):
  file_path = pathlib.Path(filename)
  if not file_path.is_file():
    print(f"Error: File {filename} does not exist.")
    return False
  if (filesize := file_path.stat().st_size) > _FILESIZE_LIMIT_IN_BYTES:
    print(f"Error: Filesize {filesize} exceeds {_FILESIZE_LIMIT_IN_BYTES}.")
    return False
  return True
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def convert_to_numpy(example):
  benchmark = {}
  example_shape = (1, _CHANNELS, _HEIGHT, _WIDTH)
  for mode in ["input", "output"]:
    benchmark[mode] = np.zeros(example_shape, dtype=np.float32)
    grid = example[mode]
    for r, _ in enumerate(grid):
      for c, color in enumerate(grid[r]):
        benchmark[mode][0][color][r][c] = 1.0
  return benchmark
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def convert_from_numpy(benchmark):
  example = []
  _, channels, height, width = benchmark.shape
  for row in range(height):
    cells = []
    for col in range(width):
      colors = [c for c in range(channels) if benchmark[0][c][row][col] == 1]
      cells.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
    while cells and cells[-1] == 10:
      cells.pop(-1)
    example.append(cells)
  while example and not example[-1]:
    example.pop(-1)
  return example
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def score_network(m):
  model = onnx_tool.loadmodel(m, {'verbose': False})
  g = model.graph
  g.graph_reorder_nodes()
  g.shape_infer(None)
  g.profile()
  if not g.valid_profile:
    print("Error: Invalid profile.")
    return None, None, None
  for key in g.nodemap.keys():
    if g.nodemap[key].op_type.upper() in _EXCLUDED_OP_TYPES:
      print(f"Error: Op type {g.nodemap[key].op_type} is not permitted.")
      return None, None, None
  return int(sum(g.macs)), int(g.memory), int(g.params)
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def load_examples(task_num):
  """Loads relevant data from ARC-AGI and ARC-GEN."""
  if not task_num:
    return _TASK_ZERO
  with open(_NEUROGOLF_DIR + f"task{task_num:03d}.json") as f:
    examples = json.load(f)
  return examples


def run_network(session, benchmark_input):
  result = session.run(["output"], {"input": benchmark_input})
  return (result[0] > 0.0).astype(float)
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def show_examples(examples, bgcolor=(255, 255, 255)):
  # Determine the dimensions of the image to be rendered.
  width, height, offset = 0, 0, 1
  for example in examples:
    grid, output = example["input"], example["output"]
    width += len(grid[0]) + 1 + len(output[0]) + 4
    height = max(height, max(len(grid), len(output)) + 4)
  # Determine the contents of the image.
  image = [[bgcolor for _ in range(width)] for _ in range(height)]
  for example in examples:
    grid, output = example["input"], example["output"]
    grid_width, output_width = len(grid[0]), len(output[0])
    for r, row in enumerate(grid):
      for c, cell in enumerate(row):
        image[r + 2][offset + c + 1] = _COLORS[cell]
    offset += grid_width + 1
    for r, row in enumerate(output):
      for c, cell in enumerate(row):
        image[r + 2][offset + c + 1] = _COLORS[cell]
    offset += output_width + 4
  # Draw the image.
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_axes([0, 0, 1, 1])
  ax.imshow(np.array(image))
  # Draw the horizontal and vertical lines.
  offset = 1
  for example in examples:
    grid, output = example["input"], example["output"]
    grid_width, grid_height = len(grid[0]), len(grid)
    output_width, output_height = len(output[0]), len(output)
    ax.hlines([r + 1.5 for r in range(grid_height+1)],
              xmin=offset+0.5, xmax=offset+grid_width+0.5, color="black")
    ax.vlines([offset + c + 0.5 for c in range(grid_width+1)],
              ymin=1.5, ymax=grid_height+1.5, color="black")
    offset += grid_width + 1
    ax.hlines([r + 1.5 for r in range(output_height+1)],
              xmin=offset+0.5, xmax=offset+output_width+0.5, color="black")
    ax.vlines([offset + c + 0.5 for c in range(output_width+1)],
              ymin=1.5, ymax=output_height+1.5, color="black")
    offset += output_width + 2
    ax.vlines([offset+0.5], ymin=-0.5, ymax=height-0.5, color="black")
    offset += 2
  ax.set_xticks([])
  ax.set_yticks([])
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def show_legend():
  image = [[(255, 255, 255) for _ in range(21)] for _ in range(5)]
  for idx, color in enumerate(_COLORS[:10]):
    image[1][2 * idx + 1] = color
  for idx, color in enumerate(_COLORS[10:]):
    for col in range(3):
      image[3][12 * idx + col + 3] = color
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_axes([0, 0, 1, 1])
  ax.imshow(np.array(image))
  for idx, _ in enumerate(_COLORS[:10]):
    color = "white" if idx in [0, 9] else "black"
    ax.text(2 * idx + 0.9, 1.1, str(idx), color=color)
  ax.text(3.4, 3.1, "no color", color="black")
  ax.text(5.75, 3.1, "<--- special colors to indicate one-hot encoding errors --->", color="black")
  ax.text(14.85, 3.1, "too many colors", color="white")
  ax.set_xticks([])
  ax.set_yticks([])
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def single_layer_conv2d_network(weight_fn, kernel_size):
  kernel_offsets = range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
  kernel_shape = [kernel_size, kernel_size]
  w_shape = [_CHANNELS, _CHANNELS, kernel_size, kernel_size]
  pads = [kernel_size // 2] * 4
  weight_cells = itertools.product(range(_CHANNELS), range(_CHANNELS),
                                   kernel_offsets, kernel_offsets)
  weights = [weight_fn(o, i, (r, c)) for (o, i, r, c) in weight_cells]

  x = onnx.helper.make_tensor_value_info("input", _DATA_TYPE, _GRID_SHAPE)
  y = onnx.helper.make_tensor_value_info("output", _DATA_TYPE, _GRID_SHAPE)
  w = onnx.helper.make_tensor("W", _DATA_TYPE, w_shape, weights)
  node_def = onnx.helper.make_node("Conv", ["input", "W"], ["output"],
                                   kernel_shape=kernel_shape, pads=pads)
  graph_def = onnx.helper.make_graph([node_def], "graph", [x], [y], [w])
  model_def = onnx.helper.make_model(graph_def, ir_version=_IR_VERSION,
                                     opset_imports=_OPSET_IMPORTS)
  return model_def
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def verify_network(network, task_num, examples):
  filename = "task{:03d}.onnx".format(task_num)
  onnx.save(network, filename)
  if not check_network(filename): return
  try:
    session = onnxruntime.InferenceSession(filename)
  except onnxruntime.ONNXRuntimeError as e:
    print(f"Error: Unable to load ONNX model: {e}")
    return
  arc_agi_right, arc_agi_wrong, arc_agi_expected = verify_subset(session, examples["train"] + examples["test"])
  arc_gen_right, arc_gen_wrong, arc_gen_expected = verify_subset(session, examples["arc-gen"])
  print(f"Results on ARC-AGI examples: {arc_agi_right} pass, {arc_agi_wrong} fail")
  print(f"Results on ARC-GEN examples: {arc_gen_right} pass, {arc_gen_wrong} fail")
  print()
  macs, memory, params = score_network(filename)
  if macs is None or memory is None or params is None:
    print("Error: Your network performance could not be measured")
  elif arc_agi_wrong + arc_gen_wrong == 0:
    print("Your network IS READY for submission!")
    print()
    print("Performance stats:")
    onnx_tool.model_profile(filename)
    points = max(1.0, 25.0 - math.log(macs + memory + params))
    print()
    print(f"It appears to require {macs} MACs + {memory} bytes + {params} params, yielding {points:.3f} points.")
    print()
    print("Next steps:")
    print(f" * Click the link below to download {filename} onto your local machine.")
    print(" * Create a zip file containing that network along with all others.")
    print(" * Submit that zip file to the Kaggle competition so that it can be officially scored.")
    print()
    display(FileLink(filename))
  else:
    print("Your network IS NOT ready for submission.")
    expected = None
    expected = arc_agi_expected if arc_agi_expected is not None else expected
    expected = arc_gen_expected if arc_gen_expected is not None else expected
    if expected is None: return
    benchmark = convert_to_numpy(expected)
    actual = {}
    actual["input"] = expected["input"]
    actual["output"] = convert_from_numpy(run_network(session, benchmark["input"]))
    print("The expected result is shown in green; your actual result is shown in red.")
    show_examples([expected], bgcolor=(200, 255, 200))
    show_examples([actual], bgcolor=(255, 200, 200))
#By https://www.kaggle.com/competitions/neurogolf-2026/data   
#neurogolf_utils.py

def verify_subset(session, example_subset):
  right, wrong, expected, error = 0, 0, None, ""
  for example in example_subset:
    benchmark = convert_to_numpy(example)
    try:
      user_output = run_network(session, benchmark["input"])
      if np.array_equal(user_output, benchmark["output"]):
        right += 1
      else:
        expected = example
        wrong += 1
    except onnxruntime.ONNXRuntimeError:
      error = traceback.format_exc()
      wrong += 1
  if error: print(f"Error: {error}")
  return right, wrong, expected
Last Golf 2025 we had:
google-code-golf-2025/code_golf_utils

And from code_golf_utils import * Change it to Neurogolf

#By Jeroen Cottaar https://www.kaggle.com/code/jeroencottaar/code-golf-simplified/notebook

task_num = 15  # Task 0 is just an illustrative example (and not eligible for points)
import sys
sys.path.append("/kaggle/input/neurogolf-2026/neurogolf_utils")
from neurogolf_utils import *
show_legend()
import sys
examples = load_examples(task_num)
show_examples(examples['train'])
show_examples(examples['test'])
for ii in range(min(10,len(examples['arc-gen'])//5)):
    show_examples(examples['arc-gen'][ii:ii+5])













Draft Session: 1h:45m.
Not a single ARC Golf 2026 graph/puzzle. Only with last Golf 2025 code.
Désolée : (
Acknowledgements:
Jeroen Cottaar https://www.kaggle.com/code/jeroencottaar/code-golf-simplified/notebook

https://www.kaggle.com/competitions/neurogolf-2026/data neurogolf_utils.py
