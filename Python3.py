import nltk
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK data (if not already downloaded)
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')

# Read Moby Dick from Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick)

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency
fdist = FreqDist(tag for (word, tag) in pos_tags)
common_pos = fdist.most_common(5)

print("5 Most Common Parts of Speech and Their Frequency:")
for pos, freq in common_pos:
    print(f"{pos}: {freq}")

from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer

# 假设您的文本数据存储在 filtered_tokens 列表中
# 这里假设 filtered_tokens 是已经过滤了停用词的单词列表

# 创建频率分布对象
fdist = FreqDist(filtered_tokens)

# 获取频率最高的前20个单词
top_20_tokens = fdist.most_common(20)

# 创建 WordNetLemmatizer 对象
lemmatizer = WordNetLemmatizer()

# 对前20个单词进行词性标注
pos_tags = pos_tag([token for token, _ in top_20_tokens])

# 对不同词性的单词进行词形还原并打印结果
print("Top 20 Tokens and Their Lemmas (Multiple POS):")
for (token, frequency), (word, pos) in zip(top_20_tokens, pos_tags):
    # 根据词性标签来指定词性进行词形还原
    if pos.startswith("N"):  # 名词
        lemma = lemmatizer.lemmatize(word, pos="n")
    elif pos.startswith("J"):  # 形容词
        lemma = lemmatizer.lemmatize(word, pos="a")
    elif pos.startswith("V"):  # 动词
        lemma = lemmatizer.lemmatize(word, pos="v")
    else:
        lemma = word  # 如果词性未知，则不进行词形还原
    
    print(f"{token} ({pos}): {lemma}")
