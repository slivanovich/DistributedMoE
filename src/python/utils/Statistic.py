from matplotlib.ticker import MultipleLocator, MaxNLocator
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Statistic:
    def __init__(
        self,
        data_filename: Optional[str],
        data_x: Optional[np.ndarray],
        data_y: Optional[np.ndarray],
        data_label: str,
        data_color: str = "blue",
        marker_type: str = "o",
        linestyle: str = "-",
    ) -> None:
        if data_x is not None and data_y is not None:
            self.data_x = data_x
            self.data_y = data_y
        elif data_filename is not None:  # do NOT use this...
            loaded_data = pd.read_csv(data_filename)
            self.data_x = loaded_data.iloc[:, 1].to_numpy()
            self.data_y = loaded_data.iloc[:, 0].to_numpy()
        assert (isinstance(self.data_x, np.ndarray) and isinstance(self.data_y, np.ndarray)) and (
            ((self.data_x.shape[0] == self.data_x.size) and (self.data_y.shape[0] == self.data_y.size))
            or (self.data_x.size == 0 and self.data_y.size == 0)
        )
        self.data_label = data_label
        self.data_color = data_color
        self.marker_type = marker_type
        self.linestyle = linestyle


def merge_statistics(
    statistics: list[Statistic],
    data_label: str,
    data_color: str = "blue",
    marker_type: str = "o",
    linestyle: str = "-",
):
    assert len(statistics) > 0

    statistics.sort(key=lambda statistic: statistic.data_x[0])  # ))

    merged = statistics[0]
    merged.data_label = data_label
    merged.data_color = data_color
    merged.marker_type = marker_type
    merged.linestyle = linestyle

    for statistic in statistics[1:]:
        merged.data_x = np.concatenate([merged.data_x, statistic.data_x], axis=0)
        merged.data_y = np.concatenate([merged.data_y, statistic.data_y], axis=0)

    return merged


class Graph:
    def __init__(
        self,
        w: int,
        h: int,
        title: str,
        x_axis_label: str,
        y_axis_label: str,
    ) -> None:
        self.width = w
        self.heigth = h
        self.title = title
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.statistics: list[Statistic] = []

    def add_statistic(self, statistic: Statistic):
        self.statistics.append(statistic)

    def show(self, graph_name, specifier: int):
        plt.figure(figsize=(self.width, self.heigth))
        plt.title(self.title)
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        for statistic in self.statistics:
            if specifier != 0:
                x_tags = []
                x_vals = []
                if specifier == 1:
                    pow = 0
                    for i in range(0, len(statistic.data_x), 1):
                        x_vals.append(2**pow)
                        if statistic.data_x[i] < 1 / 1024:
                            x_tags.append(f"{(statistic.data_x[i] * 1024**2):.0f}B")
                        elif statistic.data_x[i] >= 1 / 1024 and statistic.data_x[i] < 1:
                            x_tags.append(f"{(statistic.data_x[i] * 1024):.0f}KB")
                        elif statistic.data_x[i] >= 1 and statistic.data_x[i] < 100:
                            x_tags.append(f"{statistic.data_x[i]:.0f}MB")
                        elif statistic.data_x[i] >= 100 and statistic.data_x[i] < 1000:
                            x_tags.append(f"{statistic.data_x[i]:.0f}MB")
                        else:
                            x_tags.append(f"{(statistic.data_x[i] / 1024):.2f}GB")
                        pow += 1
                elif specifier == 2:
                    pow = 0
                    for i in range(len(statistic.data_x)):
                        x_vals.append(2**pow)
                        x_tags.append(f"{(statistic.data_x[i]):.0f}")
                        pow += 1
                plt.plot(
                    x_vals,
                    statistic.data_y,
                    label=statistic.data_label,
                    color=statistic.data_color,
                    marker=statistic.marker_type,
                    markersize=4.2,
                    linestyle=statistic.linestyle,
                )
                plt.xscale("log", base=2)
                plt.xticks(x_vals, x_tags, fontsize=6)
                ax = plt.gca()
                current_n = len(ax.get_yticks())
                ax.yaxis.set_major_locator(MaxNLocator(nbins=current_n))
            else:
                plt.plot(
                    statistic.data_x,
                    statistic.data_y,
                    label=statistic.data_label,
                    color=statistic.data_color,
                    marker=statistic.marker_type,
                    markersize=4.2,
                    linestyle=statistic.linestyle,
                )

        plt.legend()
        plt.grid(True)
        plt.savefig(f"/Users/skuralenok/arcadia/junk/skuralenok/MCTE/{graph_name}.png")
        plt.show()


def graph_test():
    g = Graph(10, 6, "test", "x_axis", "y_axis")

    x = np.arange(0, 10, 0.2)
    y = x
    g.add_statistic(Statistic(None, x, y, "linear", "black", marker_type=".", linestyle=":"))
    y = 0.2 * x * x
    g.add_statistic(Statistic(None, x, y, "square", "red", marker_type="."))
    y = np.sin(x)
    g.add_statistic(Statistic(None, x, y, "sin", "blue"))

    g.show("test", False)


# graph_test()
