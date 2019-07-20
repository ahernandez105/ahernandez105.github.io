#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Angel C. Hernandez'
SITENAME = "angel's blog"
SITEURL = ''
SITESUBTITLE = 'My name is Angel C. Hernandez and I am a graduate student at Carnegie Mellon University focusing my studies in machine learning. I am also a recipient of the Science Mathematics and Research for Transformation (SMART) Fellowship, where upon graduation I will be an engineer for the army adhereing to initiatives in warfare simulation modeling at the TRADOC Analysis Center. I have found machine learning blog posts to be an excellent resource during my academic studies and I hope mine can be a benefit to you.' 
SITEIMAGE = '/images/me.png width=300 height=200'
ICONS = (
	('github','https://github.com/ahernandez105'),
	('instagram','https://www.instagram.com/angel__christopher/'),
	('linkedin','https://www.linkedin.com/in/angel-c-hernandez/')
)


PATH = 'content'

TIMEZONE = 'America/Detroit'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
# FEED_ALL_ATOM = 'feeds/all.atom.xml'
# FEED_ALL_RSS = 'feeds/all.rss.xml'
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None

FEED_ALL_ATOM = 'feeds/all.atom.xml'
FEED_ALL_RSS = 'feeds/all.rss.xml'
AUTHOR_FEED_RSS = 'feeds/%s.rss.xml'
RSS_FEED_SUMMARY_ONLY = False

# Blogroll
#LINKS = (("colah's blog", 'https://colah.github.io/'),
#        ('distill', 'https://distill.pub/'),
#        ("Lil'Log", 'https://lilianweng.github.io/lil-log/'))

# Social widget
# SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)

DEFAULT_PAGINATION = 0

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True


STATIC_PATHS = ['images']
THEME = 'themes/pelican-alchemy/alchemy'
PLUGIN_PATHS=['/Users/angelhernandez/GitHub/ach_blog/pelican-plugins']
PLUGINS = ['render_math','pelican-cite']
PUBLICATIONS_SRC = 'content/bib.bib'