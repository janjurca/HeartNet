from utils.itkImage import ItkImage


class VolumeImage:
    def __init__(self, image: ItkImage, ax, fig, title, gl, onScroll=None) -> None:
        self.gl = gl
        self.onScroll = onScroll
        self.title = title
        self.image = image
        self.ax = ax
        self.fig = fig
        self.index = int(len(self.image.ct_scan)/2)
        self.ax_data = self.ax.imshow(self.image.ct_scan[self.index], cmap='gray')
        self.eventSetup()
        self.ax.set_title(f"{self.title} | {self.index}")

    def eventSetup(self):
        def onScroll(event):
            if self.gl.selected_axis is not self.ax:
                return

            if event.button == "up":
                self.index += 1
            if event.button == "down":
                self.index -= 1
            self.index = 0 if self.index < 0 else (len(self.image.ct_scan) - 1 if self.index > len(self.image.ct_scan) else self.index)
            self.redraw()
            if self.onScroll:
                self.onScroll(self)
        self.fig.canvas.mpl_connect('scroll_event', onScroll)

    def setIndex(self, index: int):
        self.index = index
        self.redraw()

    def redraw(self):
        self.ax.set_title(f"{self.title} | {self.index}")
        self.ax_data.set_data(self.image.ct_scan[self.index])
        self.fig.canvas.draw_idle()
