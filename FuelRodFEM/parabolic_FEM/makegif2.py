import os
import imageio.v2 as imageio  # 使用 v2 版本的 imageio

# 列出当前目录下的 PNG 文件
fnames = [fname for fname in os.listdir() if fname.endswith('.png')]

# 设置每个帧重复的次数以降低播放速度
repeat_times = 5  # 每个图片重复显示5次

with imageio.get_writer('./movie.gif', mode='I', duration=100) as writer:
    for name in fnames[::10]:  # 这里已经假设你按需选择了帧
        image = imageio.imread(name)
        # 为当前帧重复写入 repeat_times 次
        for _ in range(repeat_times):
            writer.append_data(image)