import re


def patch_affinity_svg(svg_text):
    """Patch Matplotlib SVG so that it can be read by Affinity Designer."""
    matches1 = [
        #x for x in re.finditer('font: ([0-9.]+)px ([^;"]+);', svg_text)
        x for x in re.finditer('font: oblique ([0-9.]+)px (\'DejaVu Sans\')', svg_text)
    ]
    matches2 = ([
        x for x in re.finditer('font: ([0-9.]+)px (\'DejaVu Sans\'|\'Arial\')', svg_text)
    ])
    matches3 = ([
        x for x in re.finditer('font: ([0-9.]+)px (\'Arial\')', svg_text)
    ])
    match_list = [matches2]
    for matches in match_list:
        svg_pieces = [svg_text[: matches[0].start()]]
        
        for i, match in enumerate(matches):
            # Change "font" style property to separate "font-size" and
            # "font-family" properties because Affinity ignores "font".
            font_size_px, font_family = match.groups()
            #print(match.groups())
            new_font_style = (
                f'font-size: {float(font_size_px):.1f}px; '
                f'font-family: {font_family};'
            )
            svg_pieces.append(new_font_style)
            if i < len(matches) - 1:
                svg_pieces.append(svg_text[match.end() : matches[i + 1].start()])
            else:
                svg_pieces.append(svg_text[match.end() :])
    return ''.join(svg_pieces)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help="Path to Matplotlib SVG file to patch.")
    parser.add_argument(
        '-o', '--output', help="Path to output patched file.", required=False
    )
    args = parser.parse_args()

    with open(args.fname, 'r') as f:
        svg_text = f.read()

    patched_svg = patch_affinity_svg(svg_text)

    if args.output is None:
        args.output = args.fname

    with open(args.output, 'w') as f:
        f.write(patched_svg)