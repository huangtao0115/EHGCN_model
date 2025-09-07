import csv

# 指定CSV文件路径
csv_file_path = "../data/R52/all-text.csv"
# 指定TXT文件路径
txt_file_path = "../data/R52/all-text.txt"

# 打开CSV文件
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    # 创建CSV读取器
    csv_reader = csv.reader(csv_file)

    # 打开TXT文件准备写入
    with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
        # 遍历CSV文件的每一行
        for row in csv_reader:
            # 检查行是否非空且第一列不为空
            if row and row[0]:
                # 将第一列的内容写入TXT文件，每个文本占一行
                txt_file.write(row[0] + '\n')

print("CSV to TXT conversion completed.")