# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# to print unicode characters, run this:
# export PYTHONIOENCODING=utf8

import math
import json
import os, sys
import argparse

from enum import Enum


class Direction(Enum):
    Horz = 0
    Vert = 1


class canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.vert_line_list = []
        self.horz_line_list = []
        self.characters = []

    def direction(self, line):
        Horz_Stationary = False
        Vert_Stationary = False

        if line[0][0] == line[1][0]:
            # Line Stationary in Horz Axis
            Horz_Stationary = True
        if line[0][1] == line[1][1]:
            # Line Stationary in Vert Axis
            Vert_Stationary = True

        if Horz_Stationary and Vert_Stationary:
            # Crash
            raise Exception("Line is Diagonal")
        if not Horz_Stationary and not Vert_Stationary:
            # Crash
            raise Exception("Line is a dot")
        if (not Horz_Stationary) and Vert_Stationary:
            return Direction.Horz

        if Horz_Stationary and (not Vert_Stationary):
            return Direction.Vert

    def draw_character(self, point, character):
        self.characters.append([point, character])

    def replace_character(self, point, character, replacement):
        if self.characters.count([point, character]):
            self.characters.remove([point, character])
        self.characters.append([point, replacement])

    def draw_line(self, start, finish):
        if self.direction([start, finish]) == Direction.Vert:
            self.vert_line_list.append([start, finish])
        else:
            self.horz_line_list.append([start, finish])

    def draw_square(self, center, size):
        horz_origin = math.floor((center[0] + 0.5) - (size / 2))
        horz_extent = math.ceil((center[0] + 0.5) + (size / 2) + 3)

        vert_origin = math.floor((center[1] + 0.5) - (size / 2))
        vert_extent = math.ceil((center[1] + 0.5) + (size / 2))

        top_left = (horz_origin, vert_origin)
        top_right = (horz_extent, vert_origin)
        bottom_left = (horz_origin, vert_extent)
        bottom_right = (horz_extent, vert_extent)

        self.draw_line(top_left, top_right)
        self.draw_line(top_right, bottom_right)
        self.draw_line(bottom_left, bottom_right)
        self.draw_line(top_left, bottom_left)

    def vertical_index(self, point):
        return point[1]

    def horizontal_index(self, point):
        return point[0]

    def within_line(self, point, line):
        Horz_Stationary = False
        Vert_Stationary = False

        if line[0][0] == line[1][0]:
            # Line Stationary in Horz Axis
            Horz_Stationary = True
        if line[0][1] == line[1][1]:
            # Line Stationary in Vert Axis
            Vert_Stationary = True

        # print("HS: {}, VS: {}, {}".format(Horz_Stationary, Vert_Stationary, line));

        if Horz_Stationary and Vert_Stationary:
            # Crash
            raise Exception("Line is Diagonal")
        if not Horz_Stationary and not Vert_Stationary:
            # Crash
            raise Exception("Line is a dot")

        if Horz_Stationary and (not Vert_Stationary):
            # it's a vertical line
            # Sort the tuples by Horz
            line.sort(key=self.vertical_index)
            start_line = (
                (point[1] == line[0][1])
                and (point[1] <= line[1][1])
                and (point[0] == line[0][0])
            )
            in_line = (
                (point[1] > line[0][1])
                and (point[1] < line[1][1])
                and (point[0] == line[0][0])
            )
            end_line = (
                (point[1] > line[0][1])
                and (point[1] == line[1][1])
                and (point[0] == line[0][0])
            )

            # print("Vert {}, point {}, {}".format(line, point, in_line));

        if (not Horz_Stationary) and Vert_Stationary:
            # it's a horizonal line
            line.sort(key=self.horizontal_index)
            start_line = (
                (point[0] == line[0][0])
                and (point[0] <= line[1][0])
                and (point[1] == line[0][1])
            )
            in_line = (
                (point[0] > line[0][0])
                and (point[0] < line[1][0])
                and (point[1] == line[0][1])
            )
            end_line = (
                (point[0] > line[0][0])
                and (point[0] == line[1][0])
                and (point[1] == line[0][1])
            )
            # print("Horz {}, point {}, {}".format(line, point, in_line));

        return (start_line, in_line, end_line)

    def find_horz_index(self, line_points):
        index = 0

        if line_points[1]:
            index += 2
        else:
            if line_points[0]:
                index += 1
            if line_points[2]:
                index += 4
        return index

    def transform(self, index):

        # if horz through, clear bits for horz end
        # if vert through, clear bits for vert end

        chars = {
            0: " ",
            2: "\u2500",  # horz line
            16: "\u2502",  # vert line
            9: "\u250c",  # box top left
            33: "\u2514",  # box bot left
            12: "\u2510",  # box top right
            36: "\u2518",  # box bot right
            1: "\u2576",  # right half horz line
            8: "\u2577",  # lower half vert line
            32: "\u2575",  # upper half vert line
            4: "\u2574",  # left half horz line
            18: "\u253c",  # vert AND horz
            25: "\u251c",  # vert and right
            17: "\u251c",
            20: "\u2524",  # vert and left
            10: "\u252c",  # horz and bot
            34: "\u2534",  # horz and top
            21: "\u253c",  # vert AND horz
            42: "\u253c",
        }
        try:
            char = chars[index]
        except KeyError:
            char = "x"

        return char

    def combine(self, a, b):
        return [a[0] or b[0], a[1] or b[1], a[2] or b[2]]

    def rasterize(self):
        for x in range(self.height):
            for y in range(self.width):
                char = "({},{})".format(y, x)
                index = 0
                horz_line_points = [False, False, False]
                vert_line_points = [False, False, False]

                for charloc in self.characters:
                    if charloc[0][0] == y and charloc[0][1] == x:
                        print(charloc[1], end="", sep="")
                        index = -1
                        break
                if index == 0:  # not a character, either vert or horz line
                    for line in self.horz_line_list:
                        horz_line_points = self.combine(
                            horz_line_points, self.within_line((y, x), line)
                        )
                    index += self.find_horz_index(horz_line_points)

                    for line in self.vert_line_list:
                        vert_line_points = self.combine(
                            vert_line_points, self.within_line((y, x), line)
                        )
                    index += 8 * self.find_horz_index(vert_line_points)

                    print("{}".format(self.transform(index)), end="", sep="")
                    # print(" {} ".format(index), end='', sep='')

            print("")


superscripts = {
    # 0 : u'\u2070',
    0: " ",
    1: "\u00b9",
    2: "\u00b2",
    3: "\u00b3",
    4: "\u2074",
    5: "\u2075",
    6: "\u2076",
    7: "\u2077",
    8: "\u2078",
    9: "\u2079",
}
subscripts = {
    # 0 : u'\u2080',
    0: " ",
    1: "\u2081",
    2: "\u2082",
    3: "\u2083",
    4: "\u2084",
    5: "\u2085",
    6: "\u2086",
    7: "\u2087",
    8: "\u2088",
    9: "\u2089",
}


def draw_switchbox(
    canvas,
    xoffset,
    yoffset,
    source_count,
    destination_count,
    northbound,
    southbound,
    eastbound,
    westbound,
    draw_demand=True,
    name="",
):
    c.draw_square((xoffset + 5, yoffset + 4), 2)

    # label it
    if len(name) > 0:
        c.draw_character((xoffset + 6, yoffset + 4), name[0])
    if len(name) > 1:
        c.draw_character((xoffset + 7, yoffset + 4), name[1])
    if len(name) > 2:
        c.draw_character((xoffset + 8, yoffset + 4), name[2])
    if len(name) > 3:
        c.draw_character((xoffset + 9, yoffset + 4), name[3])

    # draw source and destination count
    if source_count > 0 or destination_count > 0:
        c.draw_character((xoffset + 7, yoffset + 5), "*")

    # left of the switchbox (south)
    if northbound > 0:
        c.draw_line((xoffset + 10, yoffset + 4), (xoffset + 14, yoffset + 4))
        if draw_demand:
            c.draw_character((xoffset + 12, yoffset + 3), subscripts[northbound])
            if northbound > 6:  # if overcapacity, mark with an 'x'
                c.draw_character((xoffset + 10, yoffset + 4), "x")
                # c.draw_character((xoffset+11,yoffset+4), 'x')
                c.draw_character((xoffset + 12, yoffset + 4), "x")
    if southbound > 0:
        c.draw_line((xoffset + 0, yoffset + 5), (xoffset + 4, yoffset + 5))
        if draw_demand:
            c.draw_character((xoffset + 2, yoffset + 6), superscripts[southbound])
            if southbound > 4:  # if overcapacity, mark with an 'x'
                c.draw_character((xoffset + 1, yoffset + 5), "x")
                # c.draw_character((xoffset+2, yoffset+5), 'x')
                c.draw_character((xoffset + 3, yoffset + 5), "x")

    # below the switchbox (east)
    if eastbound > 0:
        c.draw_line((xoffset + 6, yoffset + 6), (xoffset + 6, yoffset + 8))
        if draw_demand:
            c.draw_character((xoffset + 5, yoffset + 7), superscripts[eastbound])
            if eastbound > 4:  # if overcapacity, mark with an 'x'
                c.draw_character((xoffset + 6, yoffset + 6), "x")
                # c.draw_character((xoffset+6, yoffset+7), 'x')
                c.draw_character((xoffset + 6, yoffset + 8), "x")
    if westbound > 0:
        c.draw_line((xoffset + 8, yoffset + 1), (xoffset + 8, yoffset + 3))
        if draw_demand:
            c.draw_character((xoffset + 9, yoffset + 2), superscripts[westbound])
            if westbound > 4:  # if overcapacity, mark with an 'x'
                c.draw_character((xoffset + 8, yoffset + 1), "x")
                # c.draw_character((xoffset+7, yoffset+2), 'x')
                c.draw_character((xoffset + 8, yoffset + 3), "x")


SB_WIDTH = 10
SB_HEIGHT = 5  # distances between switchboxes


def draw_switchboxes(c, switchboxes):
    for item in switchboxes:
        draw_switchbox(
            c,
            SB_WIDTH * item["row"],
            SB_HEIGHT * item["col"],
            item["source_count"],
            item["destination_count"],
            item["northbound"],
            item["southbound"],
            item["eastbound"],
            item["westbound"],
            draw_demand=True,
            name="{},{}".format(item["col"], item["row"]),
        )


# given a route, draw arrow characters to indicate the route
# route is a list of switchboxes, represented as int tuple coordinates
left_arrow = "\u2190"
up_arrow = "\u2191"
right_arrow = "\u2192"
down_arrow = "\u2193"


def draw_route(c, route):
    for i in range(len(route) - 1):
        col = route[i][0][0]
        row = route[i][0][1]
        xoffset = SB_WIDTH * row
        yoffset = SB_HEIGHT * col
        if len(route[i]) == 1:
            continue
        dirs = route[i][1]

        # draw source and destination
        if i == 0:
            c.draw_character((xoffset + 5, yoffset + 5), "S")
        if i == (len(route) - 2):
            c.draw_character((xoffset + 9, yoffset + 5), "D")

        if i == 0:
            if row == 0:  # for routes starting in the shim, draw arrows coming from PL
                c.draw_character((xoffset + 1, yoffset + 4), right_arrow)
                c.draw_character((xoffset + 2, yoffset + 4), right_arrow)
                c.draw_character((xoffset + 3, yoffset + 4), right_arrow)

        for j in range(len(dirs)):
            # draw indications for cores the route passes through
            c.replace_character((xoffset + 7, yoffset + 5), "*", "#")
            # 0 = North, 1 = East, 2 = South, 3 = West
            if dirs[j] == "North":
                c.draw_character((xoffset + 11, yoffset + 4), right_arrow)
                c.draw_character((xoffset + 12, yoffset + 4), right_arrow)
                c.draw_character((xoffset + 13, yoffset + 4), right_arrow)
                row = row + 1
            elif dirs[j] == "East":
                c.draw_character((xoffset + 6, yoffset + 7), down_arrow)
                col = col + 1
            elif dirs[j] == "South":
                c.draw_character((xoffset + 1, yoffset + 5), left_arrow)
                c.draw_character((xoffset + 2, yoffset + 5), left_arrow)
                c.draw_character((xoffset + 3, yoffset + 5), left_arrow)
                row = row - 1
            elif dirs[j] == "West":
                c.draw_character((xoffset + 8, yoffset + 2), up_arrow)
                col = col - 1
            elif dirs[j] == "DMA":
                # draw destination
                c.draw_character((xoffset + 9, yoffset + 5), "D")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw switchboxes, demands and routes")
    parser.add_argument("-j", "--json", help="Filepath for JSON file to read")
    parser.add_argument("-r", "--route_list", help="List of routes to print")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output directory. Text files of the routes will be stored here.",
    )
    args = parser.parse_args()

    if args.json:
        json_file_path = args.json
    else:
        json_file_path = "switchbox.json"  # default JSON

    with open(json_file_path) as f:
        json_data = json.load(f)

    switchboxes = []
    routes = []

    for key, item in json_data.items():
        if "switchbox" in key:
            switchboxes.append(item)
        if "route" in key:
            routes.append(item)

    max_col = 0
    max_row = 0
    for switchbox in switchboxes:
        if switchbox["col"] > max_col:
            max_col = switchbox["col"]
        if switchbox["row"] > max_row:
            max_row = switchbox["row"]

    routes_to_print = []
    if args.route_list:
        for route in args.route_list.split(","):
            routes_to_print.append(int(route.strip()))
    else:
        routes_to_print = range(len(routes))

    output_directory = "./routes/"
    if args.output:
        output_directory = args.output

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    for i in routes_to_print:
        c = canvas(12 * (max_row + 1), 5 + 5 * (max_col + 1))
        draw_switchboxes(c, switchboxes)
        filename = os.path.join(output_directory, "route{}.txt".format(i))
        sys.stdout = sys.__stdout__
        print(
            "Printing route {} of {}: {}".format(i, len(routes_to_print) - 1, filename)
        )
        with open(filename, "w") as f:
            sys.stdout = f
            print("Route {}: {}".format(i, routes[i]))
            draw_route(c, routes[i])
            c.rasterize()