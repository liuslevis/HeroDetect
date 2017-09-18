import os
train_dir = './data/input/train'
for base, dirs, files in os.walk(train_dir):
    if base != train_dir:
        print(len(files), base)

# 179 ./data/input/train/ake
# 179 ./data/input/train/bailishouyue
# 191 ./data/input/train/bailixuance
# 179 ./data/input/train/buzhihuowu
# 179 ./data/input/train/chengjisihan
# 179 ./data/input/train/diaochan
# 178 ./data/input/train/gaojianli
# 179 ./data/input/train/kai
# 179 ./data/input/train/lanlingwang
# 178 ./data/input/train/libai
# 179 ./data/input/train/liubang
# 178 ./data/input/train/makeboluo
# 177 ./data/input/train/niumo
# 179 ./data/input/train/sunbin
# 179 ./data/input/train/sunwukong
# 179 ./data/input/train/wangzhaojun
# 178 ./data/input/train/yinzheng
# 194 ./data/input/train/zhenji
# 179 ./data/input/train/zhouyu
# 179 ./data/input/train/zhuangzhou
# 179 ./data/input/train/zhugeliang
