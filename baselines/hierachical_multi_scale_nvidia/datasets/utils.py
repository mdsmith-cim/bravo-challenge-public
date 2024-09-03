import os


def make_dataset_folder(folder):
    """
    Create Filename list for images in the provided path

    input: path to directory with *only* images files
    returns: items list with None filled for mask path
    """
    items = os.listdir(folder)
    items = [(os.path.join(folder, f), '') for f in items]
    items = sorted(items)

    print(f'Found {len(items)} folder imgs')

    """
    orig_len = len(items)
    rem = orig_len % 8
    if rem != 0:
        items = items[:-rem]

    msg = 'Found {} folder imgs but altered to {} to be modulo-8'
    msg = msg.format(orig_len, len(items))
    print(msg)
    """

    return items

def get_cityscapes_colormap():
    """From cityscapes code"""
    palette = [128, 64, 128,
               244, 35, 232,
               70, 70, 70,
               102, 102, 156,
               190, 153, 153,
               153, 153, 153,
               250, 170, 30,
               220, 220, 0,
               107, 142, 35,
               152, 251, 152,
               70, 130, 180,
               220, 20, 60,
               255, 0, 0,
               0, 0, 142,
               0, 0, 70,
               0, 60, 100,
               0, 80, 100,
               0, 0, 230,
               119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    return palette