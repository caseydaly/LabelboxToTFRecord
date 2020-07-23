class Label:

    def __init__(self, xmin, xmax, ymin, ymax, label, text=""):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.label = label
        self.text = text


def label_from_labelbox_obj(top, left, height, width, label):
    ymin = top
    xmin = left
    ymax = ymin + height
    xmax = xmin + width
    return Label(xmin, xmax, ymin, ymax, label)