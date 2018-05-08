#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import codecs
import pymysql
import feedparser
import urllib
from nltk import ngrams
from hashlib import sha1
from datasketch.minhash import MinHash
from bs4 import BeautifulSoup
import time

#myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=red+tide&output=rss')
#myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=blue+green+algae&output=rss')
#myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=cyanobacteria&output=rss')
#myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=cyanohab&output=rss')
#myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=microcystin&output=rss')
#myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=algal+bloom&output=rss')
#myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=toxic+blue+green+algae&output=rss')
myfeed = feedparser.parse('https://news.google.com/news?cf=all&hl=en&pz=1&ned=us&q=clear+lake+monitor+point+blue+green+algae&output=rss')



print ("Feed title: " + myfeed['feed']['title'])
print ("Feed link: " + myfeed['feed']['link'])
print ("Feed Description: " + myfeed['feed']['description'])

#print (len(myfeed['entries']))
	
		
for post in myfeed.entries:
	# database connection
	db = pymysql.connect(host="localhost",   
                     user="root",         
                     passwd="Sem5@1234", 
                     db="sys",
					 charset='utf8mb4',
                     use_unicode=True)        
	
	
	
	# cursor object to execute database queries
	cur = db.cursor()
	
	# retrieving feed parameters into string variables
	post_title = post.title.encode('ascii', 'xmlcharrefreplace')
	post_link = post.link.encode('ascii', 'xmlcharrefreplace')
	post_desc = post.description.encode('ascii', 'xmlcharrefreplace')
	
	"""
	# Extracting current feeds entire article details
	html = urllib.urlopen(post_link).read()
	doc = html
	soup = BeautifulSoup(''.join(doc))
	link_content = soup.findAll('p')
	print type(link_content)
	print len(link_content)
	x = ''
	for link in link_content:
		post_extract = link.text.encode('ascii', 'xmlcharrefreplace')
		x += post_extract
	s = ''.join(x)
	post_detail = s.encode('ascii', 'xmlcharrefreplace')
	print ("Feed Details: "+post_detail)
	#cur.execute("INSERT INTO Feeds_Content(Feed_URL,Feed_Detail)VALUES(%s,%s)",(post_link,post_detail))
	#db.commit()
	"""
	
	# generating shingles of feed title
	n = 3
	x = []
	shingles = ngrams(post_title.split(), n)
	for grams in shingles:
		x.append(str(grams))
	s = ''.join(x)
	
	post_shingles = s.encode('ascii', 'xmlcharrefreplace')
	print ("Post Shingles: "+post_shingles)
	print type(post_shingles)
	
	# generating minHash and comparing the two before inserting in database
	
	flag_js = False
	cur.execute("SELECT Shingles FROM Feeds")
	if not cur.rowcount:
		print "Database is empty so inserting first feed"
		print cur.rowcount
		now = time.strftime("%d/%m/%Y %H:%M:%S")
		cur.execute("INSERT INTO Feeds(Title,URL,Description,Keyword,Entry_Time,Shingles)VALUES(%s,%s,%s,%s,%s,%s)",(post_title,post_link,post_desc,"Red Tide",now,post_shingles))
		db.commit()
		
		print "First row inserted"
	else:
		for m in range (int(cur.rowcount)):
			data = cur.fetchone()
			print ("Data: "+str(data))
			m1 = MinHash()
			m2 = MinHash()
			for d in post_shingles:
				m1.update(d.encode('utf8'))
			for d in str(data):
				m2.update(d.encode('utf8'))
			print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))
			if (m1.jaccard(m2)<0.8):
				continue
			else:
				print("Following article already exists in the database: ")
				flag_js = True
				break
				
		if(flag_js == False):
			print ("Ready to insert in database")
			now = time.strftime("%d/%m/%Y %H:%M:%S")
			cur.execute("INSERT INTO Feeds(Title,URL,Description,Keyword,Entry_Time,Shingles)VALUES(%s,%s,%s,%s,%s,%s)",(post_title,post_link,post_desc,"Red Tide",now,post_shingles))
			db.commit()		
		
db.commit()
db.close()