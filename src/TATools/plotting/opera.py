LASER_NAME_MAP = {
    14: "Red (Front)",
    15: "Red (Back)",
    16: "Blue",
}

PD0_NAME_MAP = {
    18: "Center [18]",
    20: "Left High-Voltage [20]",
    21: "Right High-Voltage [21]",
}

PD1_NAME_MAP = {
    17: "Forwardscatter [17]",
    19: "Backscatter [19]",
    20: "HV Left [20]",
    21: "HV Right [21]",
}

PIN_NAME_MAP = {**LASER_NAME_MAP, **PD0_NAME_MAP}

name_and_pin_str = lambda pin: f"{PIN_NAME_MAP.get(pin) or pin} [{pin}]"

VAR_NAME_MAP = {
    'integral': "Pulse Area",
    'height': 'Pulse Height',
    'laser': 'Laser',
    'pulses_per_second': "Pulses Per Second",
    'num_pulse': "#",
    'sps30_pn10': r"SPS30 $PN_{10 \mu m}$",
    'sps30_pm2p5': r"SPS30 $PM_{2.5 \mu m}$",
    'unix': "Date & Time",

    'area_25pc_width': r"Pulse Area at $\frac{1}{4}$-Peak Width",
    'area_50pc_width': r"Pulse Area at $\frac{1}{2}$-Peak Width",
    'area_75pc_width': r"Pulse Area at $\frac{3}{4}$-Peak Width",

    'raw_25pc_width': r"Pulse $\frac{1}{4}$-Peak Width",
    'raw_50pc_width': r"Pulse $\frac{1}{2}$-Peak Width",
    'raw_75pc_width': r"Pulse $\frac{3}{4}$-Peak Width",

    'imx8_temp': "Portenta Processor [I.MX8] Temp.",
    'teensy_temp': "Teensy Processor Temp.",
    'flow_temp': "Flow Temp.",
    'omb_temp_htu': "Motherboard Temp. [HTU]",
    'omb_temp_scd': "Motherboard Temp. [SCD]",
    'optical_temp0': "Laser Temp. [#1]",
    'optical_temp1': "Laser Temp. [#2]",
    'optical_temp2': "Laser Temp. [#3]",

    'max_laser_on': "Max. Initial Laser ADC Resp.",
    'height_ratio': r"Side-PD Pulse Height as % of Center-PD",
}
VAR_UNIT_MAP = {
    'integral': r"$ADC\ Count \cdot Steps$",
    'pulses_per_second': r"#/s",
    'sps30_pn10': r"#/cc",
    'sps30_pm2p5': r"$\mu g / m^3$",
    'height': 'ADC Count',
    'max_laser_on': 'ADC Count',
    'raw_25pc_width': "ADC Steps",
    'raw_50pc_width': "ADC Steps",
    'raw_75pc_width': "ADC Steps",

    'area_25pc_width': r"$ADC\ Count \cdot Steps$",
    'area_50pc_width': r"$ADC\ Count \cdot Steps$",
    'area_75pc_width': r"$ADC\ Count \cdot Steps$",

}

PD0_STYLE_MAP = {
    18: "solid",
    20: "solid",
    21: "dashed",
}
PD1_STYLE_MAP = {
    17: "solid",
    19: "dashed",
    20: "solid",
    21: "dashed",
}
LASER_COLOR_MAP = {
    16: "blue",
    14: "red",
    15: "maroon",
}

LASER_CMAP_MAP = {
    14: "Reds",
    16: "Blues",
    15: "YlOrRd"
}

var_name_or_title = lambda var_name: VAR_NAME_MAP.get(var_name) or ' '.join(var_name.split("_")).title()
var_units_or_empty = lambda var_name, prepend="": prepend + f'({VAR_UNIT_MAP.get(var_name)})' if var_name in VAR_UNIT_MAP.keys() else ""
var_name_and_units = lambda x: f"{var_name_or_title(x)}"+var_units_or_empty(x, " ")
laser_and_photodiode = lambda l, p: f"{LASER_NAME_MAP.get(l)} Laser, {PD0_NAME_MAP.get(p)} Photodiode"