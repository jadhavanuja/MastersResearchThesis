""" Runs script for each keyword every hour and loads the new feeds """

#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import codecs
import os
import time
from apscheduler.schedulers.blocking import BlockingScheduler

try:
	def feeds_extraction():
		time.sleep(5);
		print("Begin search on blue green algae keyword");
		time.sleep(5);
		os.system('python /home/acj03778/Desktop/Thesis/1-Feeds_Extraction_Scripts/BlueGreenAlgae.py');
		time.sleep(5);
		print("Begin search on cyanobacteria keyword");
		time.sleep(5);
		os.system('python /home/acj03778/Desktop/Thesis/1-Feeds_Extraction_Scripts/Cyanobacteria.py');
		time.sleep(5); 
		print("Begin search on microcystin keyword");
		time.sleep(5);
		os.system('python /home/acj03778/Desktop/Thesis/1-Feeds_Extraction_Scripts/Microcystin.py');
		time.sleep(5); 
		print("Begin search on red tide keyword");
		time.sleep(5);
		os.system('python /home/acj03778/Desktop/Thesis/1-Feeds_Extraction_Scripts/RedTide.py');
		time.sleep(5);
		print("Begin search on toxic blue green algae keyword");
		time.sleep(5);
		os.system('python /home/acj03778/Desktop/Thesis/1-Feeds_Extraction_Scripts/ToxicBlueGreenAlgae.py');
		time.sleep(5);
		print("Begin search on CyanoHAB keyword");
		time.sleep(5);
		os.system('python /home/acj03778/Desktop/Thesis/1-Feeds_Extraction_Scripts/Cyanohab.py');
		time.sleep(5)
		print("Begin content extracttion for newly added articles")
		os.system('python /home/acj03778/Desktop/Thesis/2-Article_Content_Extraction_Script/IndArticleExtract.py');
		time.sleep(5);
		print("Preprocessing newly extracted articles");
		os.system('python /home/acj03778/Desktop/Thesis/3-Article_Content_PreProcessing_Script/PreprocessArticleExtract.py');
except:
	print ("Unexpected Error: ",sys.exc_info()[0])
scheduler = BlockingScheduler()
scheduler.add_job(feeds_extraction, 'interval', minutes = 60)
scheduler.start()
