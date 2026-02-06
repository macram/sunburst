import logging
import os
import tomllib
import util

default_config_file_name = "config.toml"

class Configuration(object):
    is_default = True
    ### Parameters
    # Error margin around the detected circle
    error_margin = 10
    circle_outer_margin = 30
    min_contour_area = 5

    # Default configuration.
    ## RGB Colours
    backgroundColorUpperRange = ((230, 230, 230), (255, 255, 255))
    circleColorRangeFirst = ((0, 0, 0),(10, 10, 10))
    circleColorRangeSecond = ((0, 0, 0),(10, 10, 10))
    ink_color_first = ((0, 130, 130), (20, 255, 255))
    ink_color_second = ((160, 130, 130), (180, 255, 255))

    def __init__(self, directory):
        self.error_margin = 10
        self.circle_outer_margin = 30
        self.min_contour_area = 5
        self.backgroundColorUpperRange = ((230, 230, 230), (255, 255, 255))
        self.circleColorRangeFirst = ((0, 0, 0),(10, 10, 10))
        self.circleColorRangeSecond = ((0, 0, 0),(10, 10, 10))
        self.ink_color_first = ((0, 130, 130), (20, 255, 255))
        self.ink_color_second = ((160, 130, 130), (180, 255, 255))
        
        config_data = self.read_config_file(directory)
        
        if config_data:
            self.is_default = False
            color_config = config_data.get('colors', {})
            constants_config = config_data.get('constants', {})

            self.error_margin = constants_config.get('error_margin', self.error_margin)
            self.circle_outer_margin = constants_config.get('circle_outer_margin', self.circle_outer_margin)
            self.min_contour_area = constants_config.get('min_contour_area', self.min_contour_area)

            self.backgroundColorUpperRange = eval(color_config.get('background_color_upper_range', self.backgroundColorUpperRange))
            self.circleColorRangeFirst = eval(color_config.get('circle_color_range_first', self.circleColorRangeFirst))
            self.circleColorRangeSecond = eval(color_config.get('circle_color_range_second', self.circleColorRangeSecond))
            self.ink_color_first = eval(color_config.get('ink_color_first', self.ink_color_first))
            self.ink_color_second = eval(color_config.get('ink_color_second', self.ink_color_second))
        

    def read_config_file(self, directory):
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
