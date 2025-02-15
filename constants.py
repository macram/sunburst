### Parameters
# Error margin around the detected circle
error_margin = 10
circle_outer_margin = 30
min_contour_area = 5

# Default configuration.
## RGB Colours
backgroundColorUpperRange = ((230, 230, 230), (255, 255, 255))
circleColorUpperRange = ((0, 0, 0),(10, 10, 10))
ink_color_first = ((0, 130, 130), (20, 255, 255))
ink_color_second = ((160, 130, 130), (180, 255, 255))

default_config_file_name = "config.toml"

import logging
import os
import tomllib
import util

def open_toml_string(directory):
    content = ""
    path = directory + "/" + default_config_file_name
    if path is not None:
        if os.path.exists(path):
            with open(path, "r") as file:
                content = file.read()
                parsed_content = tomllib.loads(content)
                print(parsed_content)
    util.logger.log(logging.DEBUG, "Will use config file at " + path.__str__())
    content = tomllib.loads(content)
