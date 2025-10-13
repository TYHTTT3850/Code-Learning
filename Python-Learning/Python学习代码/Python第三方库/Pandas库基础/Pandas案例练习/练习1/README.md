1. 查看Facebook、Apple、Amazon、Netflix和Google（FAANG）股票的数据，但每只股票都被提供为单独的CSV文件。将它们合并成一个文件，并将FAANG数据的dataframe存储为faang，以便后续练习使用： 

    a) 读取aapl.csv、amzn.csv、fb.csv、goog.csv和nflx.csv文件。 

    b) 为每个dataframe添加一个名为ticker的列，表示其对应的股票代码（例如，Apple的股票代码是AAPL）；这是查找股票的方式。在这种情况下，文件名恰好是股票代码。 

    c) 将它们合并成一个dataframe。 

    d) 将结果保存为名为faang.csv的CSV文件。  

2. 使用类型转换将faang中date列的值转换为日期时间类型，并将volume列的值转换为整数。然后按日期和股票代码排序。 

3. 在faang中找到volume值最低的七行。  

4. 目前，数据处于长格式和宽格式之间。使用melt()将其转换为完全的长格式。提示：date和ticker是我们标识变量（它们唯一标识每一行）。我们需要熔化其余的列，以避免为open、high、low、close和volume创建单独的列。

5. 假设发现2018年7月26日数据记录过程中存在错误。应如何处理？ 

6. 欧洲疾病预防控制中心（ECDC）提供了一个关于新冠状病毒病例的公开数据集，名为“全球按国家划分的新冠状病毒新报告病例数”（https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide）。
我们使用一个数据子集，包含2020年1月1日至2020年9月18日的数据。清理并转换数据，使其处于宽格式： 

    a) 读取covid19_cases.csv文件。 

    b) 使用dateRep列中的数据和pd.to_datetime()函数创建一个日期列。 

    c) 将日期列设为索引并按索引排序。 

    d) 将所有“United_States_of_America”替换为“USA”，将所有“United_Kingdom”替换为“UK”。提示：可以对整个dataframe运行replace()方法。 

    e) 使用countriesAndTerritories列，将清理后的新冠状病毒病例数据过滤为阿根廷、巴西、中国、哥伦比亚、印度、意大利、墨西哥、秘鲁、俄罗斯、西班牙、土耳其、英国和美国。 

   f) 转换数据，使索引包含日期，列包含国家名称，值为病例数（cases列）。确保用0填充NaN值
