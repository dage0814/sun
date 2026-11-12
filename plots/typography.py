import matplotlib as mpl
from matplotlib.text import Text


def configure_serif_fonts():
    mpl.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'],
        'font.serif': ['Times New Roman', 'SimSun'],
        'font.weight': 'normal', 'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'mathtext.bfit': 'Times New Roman:italic:bold',
        'mathtext.cal': 'Times New Roman',
        'mathtext.sf': 'Times New Roman',
        'mathtext.tt': 'Times New Roman',
        'mathtext.fallback': None,
    })


def calibrate_print_text(fig, *, pad_inches=0.1, width_mm=150, target_pt=10.5):
    configure_serif_fonts()
    print_width_in = width_mm / 25.4
    source_pt = target_pt
    for _ in range(30):
        fig.canvas.draw()
        texts = [text for text in fig.findobj(match=Text)
                 if text.get_text().strip()]
        for text in texts:
            text.set_fontfamily(['Times New Roman', 'SimSun'])
            text.set_fontweight('normal')
            text.set_math_fontfamily('custom')
            text.set_fontsize(source_pt)
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        export_width = bbox.width + 2 * pad_inches
        updated_pt = target_pt * export_width / print_width_in
        if abs(updated_pt - source_pt) < 0.0005:
            break
        source_pt = updated_pt
    printed_pt = source_pt * print_width_in / export_width
    if abs(printed_pt - target_pt) > 0.015:
        raise RuntimeError(f'Print-size calibration failed: {printed_pt:.4f} pt')
    result = {'printedWidthMM': width_mm, 'targetPrintedTextPt': target_pt,
              'sourceTextPt': source_pt, 'actualPrintedTextPt': printed_pt,
              'exportWidthInches': export_width,
              'printedHeightMM': (bbox.height + 2 * pad_inches) / export_width * width_mm,
              'textCount': len(texts), 'mathSubscripts': 'relative mathtext sizing preserved',
              'latinFont': 'Times New Roman', 'chineseFont': 'SimSun',
              'ordinaryWeight': 'normal', 'mathFont': 'Times New Roman custom'}
    fig._print_typography = result
    return result

