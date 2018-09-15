import cv2
import xlrd
import numpy

def produce():
    sheet = xlrd.open_workbook('Distribution.xlsx')
    sheet = sheet.sheets()[0]
    