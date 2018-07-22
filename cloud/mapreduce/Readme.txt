在本例中，训练集是test_max文件中的数据。
data是测试样例。
MyMapper.class：读入训练集文件test_max，对于每个测试集样本，求出该测试集样本到所有训练集样本的距离并确定前k个最近的训练样本标签。将测试集样本distance作为key，key作为value输出。
MyReducer.class：求出k个训练样本标签中出现最多的id，并将其作为测试样本的预测标签。将测试样本+预测标签写到输出文件中。
Distance.class：求两个向量之间的欧拉距离。
Element.class：将训练样本字符串分为属性和标签。