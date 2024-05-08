import os
import imageio.v2 as imageio  # 使用 v2 版本的 imageio

# 列出当前目录下的 PNG 文件
fnames = [fname for fname in os.listdir() if fname.endswith('.png')]

with imageio.get_writer('./movie.gif', mode='I') as writer:
    for name in fnames[::10]:
        image = imageio.imread(name)
        writer.append_data(image)
