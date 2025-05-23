# **Page rendering**
page_width = 1700
page_height = 2400

page_size = (page_width, page_height)

output_format = ".png"

boundary_width = 10

# **Font coverage**
# How many characters of the dataset should the font files support
font_character_coverage = 0.80


# **Panel Drawing**
# *Panel ratios*

# TODO: Figure out page type distributions
num_pages_ratios = {
    1: 0.146,
    2: 0.189,
    3: 0.135,
    4: 0.189,
    5: 0.189,
    6: 0.152,
}

vertical_horizontal_ratios = {
    "v": 0.075,
    "h": 0.075,
    "vh": 0.85
}

# Panel transform chance

panel_transform_chance = 0.90

# Panel shrinking

panel_shrink_amount = -25

# Panel removal

panel_removal_chance = 0.0
panel_removal_max = 2

# Background adding
background_add_chance = 0.0

# **Speech bubbles**
max_speech_bubbles_per_panel = 2
bubble_to_panel_area_max_ratio = 0.4
bubble_mask_x_increase = 15
bubble_mask_y_increase = 15
min_font_size = 54
max_font_size = 72

TEXT_SETTINGS = {
    "enabled": False,
    # or text probability = 0
}

# 말풍선만 활성화하기 위한 설정
SPEECH_BUBBLE_SETTINGS = {
    "enabled": True,  # 말풍선 활성화 여부
    "min_per_panel": 0,  # 패널당 최소 말풍선 수
    "max_per_panel": 1,  # 패널당 최대 말풍선 수
}

# *Transformations*

# Slicing
double_slice_chance = 0.25
slice_minimum_panel_area = 0.2
center_side_ratio = 0.7

# Box transforms
box_transform_panel_chance = 0.1
panel_box_trapezoid_ratio = 0.5

# How much at most should the trapezoid/rhombus start from
# as a ratio of the smallest panel's width or height
trapezoid_movement_limit = 50
rhombus_movement_limit = 50

full_page_movement_proportion_limit = 25
