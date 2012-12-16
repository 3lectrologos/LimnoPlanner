import os
import sys
import unittest


top = os.getcwd()
sys.path.append(os.path.dirname(top))
suite = unittest.TestLoader().discover('.', pattern='*_test.py')
unittest.TextTestRunner().run(suite)
