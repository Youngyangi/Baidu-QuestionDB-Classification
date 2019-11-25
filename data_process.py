import pandas as pd
import jieba
import os.path
import config


def text_process(strings):
    cut = []
    for string in strings:
        string = string.replace('\n', '')
        cut.append(' '.join(jieba.cut(string)))
    return cut


gd = pd.read_csv(os.path.join(config.history_source_dir, '古代史.csv'))
xd = pd.read_csv(os.path.join(config.history_source_dir, '现代史.csv'))
jd = pd.read_csv(os.path.join(config.history_source_dir, '近代史.csv'))

gd_data = pd.Series(text_process(gd['item']), name='question')
gd_tag = pd.Series(['gd']*len(gd_data), name='tag')
xd_data = pd.Series(text_process(xd['item']), name='question')
xd_tag = pd.Series(['xd']*len(xd_data), name='tag')
jd_data = pd.Series(text_process(jd['item']), name='question')
jd_tag = pd.Series(['jd']*len(jd_data), name='tag')
gd_ = pd.concat([gd_data, gd_tag], axis=1)
xd_ = pd.concat([xd_data, xd_tag], axis=1)
jd_ = pd.concat([jd_data, jd_tag], axis=1)
df = pd.concat([gd_, xd_, jd_], ignore_index=True)
df = pd.DataFrame(df)
print(type(df))
print(type(gd))
df.to_csv(os.path.join(config.output_dir, 'history.csv'), encoding='utf8', index=False)







