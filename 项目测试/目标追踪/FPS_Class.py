#!/usr/bin/env python3
# coding:utf-8
"""
# File       : FPS_Class.py
# Time       ：10/24/22 1:09 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：FPS类，用于计算FPS
"""
import datetime


class FPS:
    def __init__(self):
        self._star = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._star = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._star).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()

