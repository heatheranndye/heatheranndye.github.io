AUTHOR = 'heatheranndye'
SITENAME = 'Python, data science, and art'
SITEURL = 'https://heatheranndye.github.io/'
PATH = 'content'
THEME ="Flex2"
THEME_COLOR ="light"
TIMEZONE = 'US/Central'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

DEFAULT_CATEGORY = 'Data Science'
DISPLAY_PAGES_ON_MENU = True

MARKUP = ("md")


LIQUID_TAGS = ["img", "include_code","notebook"]

#from pelican_jupyter import liquid as nb_liquid

# IPYNB_MARKUP_USE_FIRST_CELL = True

IGNORE_FILES = [".ipynb_checkpoints"]
STATIC_PATHS =["images","notebook","code"]

TAGS_SAVE_AS ='tags.html'
CATEGORIES_SAVE_AS ='categories.html'

# DISPLAY_CATEGORIES_ON_MENU = False
# #DISPLAY_PAGES_ON_MENU = False

MENUITEMS = (
    ("Archives", "/archives.html"),
    ("Categories", "/categories.html"),
    ("Tags", "/tags.html"),
)

# USE_FOLDER_AS_CATEGORY = False
MAIN_MENU = True
# HOME_HIDE_TAGS = True

#from io import open

#EXTRA_HEADER = open('_nb_header.html', encoding='utf-8').read()

# Blogroll
#LINKS = (('Pelican', 'https://getpelican.com/'),
#        ('Python.org', 'https://www.python.org/'),
#        ('You can modify those links in your config file', '#'),)
LINKS = (('Introduction', '/blog-1.html'),)
# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True