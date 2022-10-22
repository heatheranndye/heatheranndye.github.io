AUTHOR = 'heatheranndye'
SITENAME = 'Python, data science, and art'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'US/Central'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

MARKUP = ("md")





LIQUID_TAGS = ["img", "literal", "video", "youtube",
               "vimeo", "include_code","notebook"]
IGNORE_FILES = [".ipynb_checkpoints"]
STATIC_PATHS =["images","notebook"]

#from io import open

#EXTRA_HEADER = open('_nb_header.html', encoding='utf-8').read()

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True