* 这个文件夹是我准备的一些可以分析的数据和表格


0.0 test.csv是官方测试集，里面没有预测结果。

0.1 train.csv是官方给的训练集

0.2 test_labels 没啥用我感觉

0.3 HANK_MNB是主要用来提交代码的文件，这个文件可以直接输出符合标准的submission。使用100%的train来训练模型，使用test.csv来预测结果。

0.4 MNB_train&test 是用来产生分类报告的，跟HANK_MNB的区别在于，这个文件将train分成了80%为训练集，20%当测试集


1. 观察数据组成：见 looking at the data

2. comment的长度统计：见comments length

3. 对于class label的描述和统计表格： 见label stat和label_stat_description

4. 我首先使用的是BNB但是效果很差，accuracy只有0.85

5. 之后我使用了MNB，使用countVectorizer，调整max feature得到最优结果，见 ：MNB countvec max feature 调参数 图表

6.之后在MNB的基础上，我是用tfidfVectorizer，调整max feature得到最优结果，见 ：MNB tfidf max feature 调参 图表

7. MNB 使用 tfidf的效果明显优于countvec，所以使用tfidf （使用应用stopwords removing的 最终分数是 0.95452 0.95349）

8. 我用 MNB_train&test.py 这个文件将train 集分成 80%为训练，20%为测试，得出在 MNB，tfidf，max feature为3750的情况下的分类报告，放在了MNB_classfication_report里面

9. 我额外使用了nlp process来代替stopwords removing，但是结果反而下降，所以不采用。

10. 关于模型性能的问题，可以等alex写出来他的模型。来横向对比