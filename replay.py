#! /usr/bin/env python

import plan
import argparse


parser = argparse.ArgumentParser(description='Replay recorded path.')
parser.add_argument('fpath', help='log file path')
args = parser.parse_args()
rec = plan.Recorder.from_file(args.fpath)
rec.replay()
