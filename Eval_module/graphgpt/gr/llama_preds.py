import numpy as np
import re
from core.data_utils.load import load_data
import csv
from tqdm import tqdm
from sklearn.metrics import f1_score


# dataset = 'pubmed'
dataset = 'chemistry'
data, num_classes, text = load_data(dataset, use_text=True, use_gpt=True)

if dataset == 'chemistry':
    classes = [
        'ENGINEERING', 'MATERIALSCIENCE', 'PHYSICS', 'CHEMISTRY', 'COMPUTERSCIENCE', 'MEDICINE', 'AGRICULTURE', 'MATHEMATICS', 'PUBLIC', 'GEOSCIENCES', 
        'EDUCATION', 'DENTISTRY', 'RADIOLOGY', 'HUMANITIES', 'ELECTROCHEMISTRY', 'NANOSCIENCE&NANOTECHNOLOGY', 'ENVIRONMENTALSCIENCES', 'ENERGY&FUELS', 
        'METALLURGY&METALLURGICALENGINEERING', 'GREEN&SUSTAINABLESCIENCE&TECHNOLOGY', 'WATERRESOURCES', 'POLYMERSCIENCE', 'BIOPHYSICS', 'BIOTECHNOLOGY&APPLIEDMICROBIOLOGY', 
        'INSTRUMENTS&INSTRUMENTATION', 'MULTIDISCIPLINARYSCIENCES', 'BIOCHEMISTRY&MOLECULARBIOLOGY', 'CRYSTALLOGRAPHY', 'OPTICS', 'SPECTROSCOPY', 
        'BIOCHEMICALRESEARCHMETHODS', 'FOODSCIENCE&TECHNOLOGY', 'ACOUSTICS', 'TOXICOLOGY', 'THERMODYNAMICS', 'METEOROLOGY&ATMOSPHERICSCIENCES', 
        'MINERALOGY', 'BIOLOGY', 'NUCLEARSCIENCE&TECHNOLOGY', 'MICROSCOPY', 'PHARMACOLOGY&PHARMACY', 'AGRICULTURALENGINEERING', 'MECHANICS', 
        'CONSTRUCTION&BUILDINGTECHNOLOGY', 'MINING&MINERALPROCESSING', 'MARINE&FRESHWATERBIOLOGY', 'QUANTUMSCIENCE&TECHNOLOGY', 'LIMNOLOGY', 
        'MICROBIOLOGY', 'NUTRITION&DIETETICS', 'GEOCHEMISTRY&GEOPHYSICS', 'ENVIRONMENTALSTUDIES', 'PLANTSCIENCES', 'MATHEMATICAL&COMPUTATIONALBIOLOGY', 
        'AGRONOMY', 'ENDOCRINOLOGY&METABOLISM', 'TRANSPORTATIONSCIENCE&TECHNOLOGY', 'SOILSCIENCE', 'CELLBIOLOGY', 'ONCOLOGY', 'GENETICS&HEREDITY', 
        'FORESTRY', 'INFECTIOUSDISEASES', 'IMMUNOLOGY', 'MATHEMATICS', 'ARCHAEOLOGY', 'AUTOMATION&CONTROLSYSTEMS', 'ASTRONOMY&ASTROPHYSICS', 'ECOLOGY', 
        'ART', 'DERMATOLOGY', 'TRANSPLANTATION', 'HORTICULTURE', 'VIROLOGY', 'PHYSIOLOGY', 'EVOLUTIONARYBIOLOGY', 'MEDICALINFORMATICS', 'ALLERGY', 
        'ENTOMOLOGY', 'GASTROENTEROLOGY&HEPATOLOGY', 'ROBOTICS', 'SURGERY', 'ANTHROPOLOGY', 'OCEANOGRAPHY', 'VETERINARYSCIENCES', 'NEUROSCIENCES', 
        'INFORMATIONSCIENCE&LIBRARYSCIENCE', 'ANATOMY&MORPHOLOGY', 'INTEGRATIVE&COMPLEMENTARYMEDICINE', 'INTERNATIONALRELATIONS', 'STATISTICS&PROBABILITY', 
        'LOGIC', 'MYCOLOGY', 'PARASITOLOGY', 'ECONOMICS', 'ARCHITECTURE', 'TRANSPORTATION', 'MEDICALLABORATORYTECHNOLOGY', 'UROLOGY&NEPHROLOGY', 
        'ZOOLOGY', 'CLINICALNEUROLOGY', 'CELL&TISSUEENGINEERING', 'OPHTHALMOLOGY', 'IMAGINGSCIENCE&PHOTOGRAPHICTECHNOLOGY', 'TELECOMMUNICATIONS', 
        'FISHERIES', 'NOTHING'
    ]
escaped_classes = []
for cls in classes:
    if '(' in cls and ')' in cls:
        main_part, bracket_part = cls.split(' (')
        bracket_part = bracket_part.rstrip(')')
        # 匹配六种可能的情况，包括连字符和原始空格
        escaped_cls = f"(?:{re.escape(main_part)} \\({re.escape(bracket_part)}\\)|{re.escape(bracket_part)} \\({re.escape(main_part)}\\)|{re.escape(main_part)}|{re.escape(bracket_part)}|{re.escape(main_part.replace(' ', '-'))}|{re.escape(main_part)})"
    else:
        # 包括连字符和原始空格的情况
        escaped_cls = f"(?:{re.escape(cls)}|{re.escape(cls.replace(' ', '-'))})"
    escaped_classes.append(escaped_cls)

classes_regex = '(' + '|'.join(escaped_classes) + ')'
pred = []
cnt = 0
class_map = {}
for i, cls in enumerate(classes):
    if '(' in cls and ')' in cls:
        main_part, bracket_part = cls.split(' (')
        bracket_part = bracket_part.rstrip(')')
        class_map[f"{main_part} ({bracket_part})".lower()] = i
        class_map[f"{bracket_part} ({main_part})".lower()] = i
        class_map[main_part.lower()] = i
        class_map[bracket_part.lower()] = i
        class_map[main_part.replace(' ', '-').lower()] = i  # 添加连字符版本
    else:
        class_map[cls.lower()] = i
        class_map[cls.replace(' ', '-').lower()] = i  # 添加连字符版本
for p in tqdm(text):
    # 从 "Answer" 到 "Answer" 后的第一个句号之间的部分
    answer_section = re.search(r'\n\nAnswer(.*?)\n\nExplanation', p, re.DOTALL) or re.search(r'Answer(.*?\.)', p, re.DOTALL) or re.search(r'\n \n Answer(.*?)\n\nExplanation', p, re.DOTALL) or re.search(r'Answer(.*?)\n\nExplanation', p, re.DOTALL)
    if answer_section:
        tp = answer_section.group(1)
    else:
        tp = ""
    matches = re.findall(classes_regex, tp.strip(), re.IGNORECASE)
    mapped = [class_map[m.lower()] for m in matches if m.lower() in class_map]
    if len(mapped) == 0:
    # 从 "Answer" 到 "Answer" 后的第二个句号之间的部分
        answer_section = re.search(r'Answer(.*?\.\s*.*?\.)', p, re.DOTALL)
        if answer_section:
            p = answer_section.group(1)
        else:
            p = ""
        matches = re.findall(classes_regex, p.strip(), re.IGNORECASE)
        mapped = [class_map[m.lower()] for m in matches if m.lower() in class_map]
    if len(mapped) == 0:
        # print("EMPTY: ", p)
        mapped = [1]
        cnt += 1
    pred.append(mapped)

first_pred = [p[0] for p in pred]

labels = data.y.squeeze()
acc = (labels.numpy() == first_pred).sum()/len(labels)
f1_macro = f1_score(labels.numpy(), first_pred, average='macro')
f1_micro = f1_score(labels.numpy(), first_pred, average='micro')
f1_weighted = f1_score(labels.numpy(), first_pred, average='weighted')

print(f'Acurracy: {acc:.4f}')
print(f'F1 Macro: {f1_macro:.4f}')
print(f'F1 Micro: {f1_micro:.4f}')
print(f'F1 Weighted: {f1_weighted:.4f}')
# with open(f'{dataset}.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for item in pred:
#         writer.writerow(item)