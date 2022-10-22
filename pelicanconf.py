AUTHOR = 'heatheranndye'
SITENAME = 'Python, data science, and art'
SITEURL = ''
PATH = 'content'
THEME ="Flex-e63fdae267319fdfb5a0788fe2de9e75ce063569"

TIMEZONE = 'US/Central'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

MARKUP = ("md")




LIQUID_TAGS = ["img", "include_code","notebook"]

#from pelican_jupyter import liquid as nb_liquid

# IPYNB_MARKUP_USE_FIRST_CELL = True

IGNORE_FILES = [".ipynb_checkpoints"]
STATIC_PATHS =["images","notebook","code"]



MENUITEMS = (
    ("Archives", "/archives.html"),
    ("Categories", "/categories.html"),
    ("Tags", "/tags.html"),
)


#from io import open

#EXTRA_HEADER = open('_nb_header.html', encoding='utf-8').read()

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True