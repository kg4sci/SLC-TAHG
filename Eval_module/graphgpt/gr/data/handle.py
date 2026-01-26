import json

# 设置你的文件路径
file_path = 'arxiv_pub_node_st_cot_link_mix.json'
output_file_path = 'arxiv_pub_node_st_cot_link_mix_1000.json'  # 你想保存的新文件路径

# 读取JSON文件并仅保存前1000行（假设文件是数组或列表形式）
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 加载整个JSON文件

    # 如果数据是一个列表或字典，根据你的文件结构调整
    # 这里假设文件数据是一个大列表，取前1000个元素
    preview_data = data[:1000] if isinstance(data, list) else list(data.items())[:1000]

# 将前1000行保存到新文件中
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(preview_data, output_file, indent=4, ensure_ascii=False)

print(f"前1000行数据已保存到 {output_file_path}")
