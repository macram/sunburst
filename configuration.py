import logging
import os
import tomllib
import util

default_config_file_name = "config.toml"

class Configuration(object):
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

    @staticmethod
    def default(self):
        error_margin = 10
        circle_outer_margin = 30
        min_contour_area = 5
        backgroundColorUpperRange = ((230, 230, 230), (255, 255, 255))
        circleColorUpperRange = ((0, 0, 0),(10, 10, 10))
        ink_color_first = ((0, 130, 130), (20, 255, 255))
        ink_color_second = ((160, 130, 130), (180, 255, 255))

    @staticmethod
    def read_config_file(directory):
        content = ""
        path = directory + "/" + default_config_file_name
        if path is not None:
            if os.path.exists(path):
                with open(path, "r") as file:
                    util.logger.log(logging.DEBUG, "Will use config file at " + path.__str__())
                    content = file.read()
                    parsed_content = tomllib.loads(content)
                    print(parsed_content)
        content = tomllib.loads(content)
        return content

default_configuration = Configuration.default