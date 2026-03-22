#!/usr/bin/env python3
"""
wpd_to_csv.py — Convert WebPlotDigitizer JSON exports to PI curve CSV.

Usage:
    python data/wpd_to_csv.py data/wpd_LV1994_25C.json data/wpd_LV1994_35C.json ...

Naming convention for input files:
    wpd_{SOURCE}_{TEMP}C.json
    e.g., wpd_LV1994_25C.json  →  source=LV1994, temp=25

Output:
    data/spirulina_pi_temperature.csv (appends/creates)

Author: M. Iizumi & T. Iizumi (Miosync, Inc.)
"""
import json
import csv
import sys
import os
import re

def parse_filename(path):
    """Extract source_id and temperature from filename."""
    basename = os.path.splitext(os.path.basename(path))[0]
    # Expected: wpd_SOURCE_TEMPc or wpd_SOURCE_TEMPC
    m = re.match(r'wpd_(.+?)_(\d+)[Cc]', basename)
    if m:
        return m.group(1), int(m.group(2))
    return basename, None

def load_wpd_json(path):
    """Load and parse WebPlotDigitizer v4 JSON export."""
    with open(path, 'r') as f:
        data = json.load(f)

    points = []
    for dataset in data.get('datasetColl', []):
        for pt in dataset.get('data', []):
            # WPD v4 format: {"x": float, "y": float} or [x, y]
            if isinstance(pt, dict):
                x, y = pt.get('x', pt.get('value', [0,0])[0]), pt.get('y', pt.get('value', [0,0])[1])
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = pt[0], pt[1]
            else:
                continue
            points.append((float(x), float(y)))

    # Sort by irradiance
    points.sort(key=lambda p: p[0])
    return points

def main():
    if len(sys.argv) < 2:
        print("Usage: python wpd_to_csv.py <wpd_file1.json> [wpd_file2.json ...]")
        sys.exit(1)

    output_path = os.path.join(os.path.dirname(sys.argv[1]), 'spirulina_pi_temperature.csv')

    rows = []
    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue

        source_id, temp_c = parse_filename(path)
        points = load_wpd_json(path)

        print(f"  {path}: source={source_id}, T={temp_c}°C, {len(points)} points")

        for irr, prate in points:
            rows.append({
                'source_id': source_id,
                'species': 'S_platensis',
                'temp_C': temp_c if temp_c else '',
                'irradiance': f"{irr:.1f}",
                'P_gross': f"{prate:.4f}",
                'P_net': '',
                'method': 'O2_electrode',
                'notes': f'digitized_from_{source_id}'
            })

    # Write CSV
    fieldnames = ['source_id', 'species', 'temp_C', 'irradiance',
                  'P_gross', 'P_net', 'method', 'notes']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} data points to {output_path}")

if __name__ == '__main__':
    main()
