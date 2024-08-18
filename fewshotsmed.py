COT = """Relevant Context: The Nile River is the longest river in the world, spanning approximately 6,650 kilometers (4,132 miles) in length. It flows through eleven countries in northeastern Africa, including Egypt, Sudan, and Uganda.
Question: What is the longest river in the world?
Thought: The question asks for the longest river in the world, which I know is the Nile River based on the context provided.
Action: Finish[Nile River]

Relevant Context: Ludwig van Beethoven was a German composer and pianist who was a crucial figure in the transition between the Classical and Romantic eras in Western classical music. One of his most famous compositions is the Symphony No. 9, also known as the "Choral" symphony.
Question: Which composer created Symphony No. 9?
Thought: The question is asking for the composer of Symphony No. 9. Based on the context, I know that Ludwig van Beethoven composed this symphony.
Action: Finish[Ludwig van Beethoven]

Relevant Context: Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy. During this process, plants absorb sunlight, carbon dioxide, and water to produce glucose and oxygen.
Question: What do plants produce during photosynthesis?
Thought: The question is asking about the products of photosynthesis. From the context, I know that plants produce glucose and oxygen during this process.
Action: Finish[Glucose and Oxygen]
"""

COT_REFLECT = """
Relevant Context: Ernest Hemingway's novel "The Old Man and the Sea" tells the story of Santiago, an aging Cuban fisherman, who struggles to catch a giant marlin in the Gulf Stream. The book won the Pulitzer Prize for Fiction in 1953 and contributed to Hemingway's Nobel Prize for Literature in 1954.
Question: Which literary award did "The Old Man and the Sea" contribute to Hemingway winning?
Thought: The question is asking which award "The Old Man and the Sea" contributed to Hemingway winning. Based on the context, I know the novel won the Pulitzer Prize for Fiction and contributed to his Nobel Prize for Literature.
Action: Finish[Pulitzer Prize for Fiction]

Reflection: My answer was correct based on the context, but may not be the exact answer stored by the grading environment. Next time, I should try to provide a less verbose answer like "Pulitzer Prize" or "Nobel Prize."

Context: On 14 October 1947, Chuck Yeager, a United States Air Force test pilot, became the first person to break the sound barrier by flying the Bell X-1 experimental aircraft at an altitude of 45,000 feet.
Charles Elwood "Chuck" Yeager (13 February 1923 - 7 December 2020) was a United States Air Force officer, flying ace, and test pilot. He is best known for becoming the first person to break the sound barrier, which he achieved in the Bell X-1 aircraft named Glamorous Glennis. Yeager was also a distinguished fighter pilot during World War II and was credited with shooting down at least 12 enemy aircraft. In 1973, he was inducted into the National Aviation Hall of Fame for his significant contributions to aviation.
Question: Who is the first person to break the sound barrier?
Thought: The question is asking for the first person to break the sound barrier. From the context, I know that Chuck Yeager, a United States Air Force test pilot, was the first person to break the sound barrier.
Action: Finish[Chuck Yeager]

Reflection: Upon reflecting on the incorrect answer I provided, I realize that I may not have given the full name of the individual in question. In the context, both the given name and the nickname were mentioned, and I only used the nickname in my response. This could have been the reason my answer was deemed incorrect. Moving forward, when attempting this question again or similar questions, I will make sure to include the complete name of the person, which consists of their given name, any middle names, and their nickname (if applicable). This will help ensure that my answer is more accurate and comprehensive."""
# TODO: 20240818ver初次编辑MEDCOT
# TODO: token减少优化
# 在含有evidence的样本中，使用context字数最少、中等，最多的3例作为COT标杆。他们的行数分别为96,130,113
# context作为Relevant Context，预处理的Question作为Question，evidence作为Thought，结果作为Action

MED_COT="""Relevant Context: 入院情况：1.老年男性，慢性病程；2.半年前无明显诱因出现上腹部隐痛不适，阵发性，可耐受，时有胀痛，有恶心，未吐，伴有大便习惯及性状改变，表现为3-7天排便一次，时有腹泻、便秘交替进行，无粘液脓血便，无呕血、黑便，无发热，曾行腹部超声检查提示脂肪肝，胆囊腺肌症，为求进一步治疗，收入我科。 主诉：间断上腹部疼痛半年。 现病史：患者半年前无明显诱因出现上腹部隐痛不适，阵发性，可耐受，时有胀痛，有恶心，未吐，伴有大便习惯及性状改变，表现为3-7天排便一次，时有腹泻、便秘交替进行，无粘液脓血便，无呕血、黑便，无发热，曾行腹部超声检查提示脂肪肝，胆囊腺肌症，为求进一步治疗，收入我科。病来精神状态可，无消瘦、乏力，无排尿正常，体重无增减。 既往史: 否认高血压、心脏病史，否认糖尿病、脑血管病、精神疾病史。30余年前肺结核病史，已治愈。否认肝炎史、疟疾史。手术史：有，于2018年04月28日因中耳炎全麻下行左侧完壁式乳突根治+后鼓室开放+鼓室成型术（II型）。过敏史：无，输血史：无，预防接种史：无，传染病史：无。其他系统回顾无特殊。 个人史：出生原籍，久居于本地，否认疫水、疫区接触史，否认其他放射性物质及毒物接触史。免疫接种史不详。否认吸烟史，有饮酒史，否认去过疫区、否认接触过从疫区来的人。饮酒史40余年，42-56°白酒3两/天；未戒酒。 家族史：父母已逝，兄弟姐妹9人，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。 检查分类: 超声 检查项目名称: 病房血管彩超 检查部位:病房颈动脉彩超 检查所见: 颈动脉:  双侧颈总动脉、颈内动脉、颈外动脉可视段管径对称，内-中膜不均匀增厚，左侧颈总动脉分叉处后壁见大小约13.9x4.0mm等回声斑块，管腔未见明显狭窄和局限性扩张，管腔内可见血流充盈，斑块处血流充盈缺损，血流速度正常。 检查结论: 双侧颈动脉内-中膜不均匀增厚伴左侧斑块形成 生化检查: 标本质量-静脉血(空腹)、总前列腺特异性抗原(TPSA)-静脉血(空腹)：4.43ng/mL、备注-静脉血(空腹)、游离前列腺特异性抗原(FPSA)-静脉血(空腹)：0.73ng/mL
Question: 请全程使用中文进行回答。请使用你的已有知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。
Thought: 患者老年男性，慢性病程；长期大量饮酒史，超声提示脂肪肝。
Action: Finish[酒精性肝病]


Relevant Context: 入院情况：患者老年女性，慢性病程；主要临床表现为乏力、食欲减退及恶心，查肝生化示ALT、AST、GGT 明显升高，ALB降低，IgG升高，ANA、SMA阳性，于我院行肝脏穿刺等检查诊断“自身免疫性肝炎”明确，规律服用激素+硫唑嘌呤，目前激素已停用，硫唑嘌呤维持治疗。2年前发现肿瘤标记物异常，胃镜、结肠镜检查考虑“胃息肉（山田I型），结肠多发息肉（山田I-IV型）”。此次为复查入院。 主诉：间断乏力6年，发现肿瘤标记物异常2年。 现病史：患者6年前因乏力、食欲减退及恶心就诊于我科，发现肝功能异常，查肝生化示ALT、AST、GGT 明显升高，ALB降低，IgG 1870mg/dl，ANA+1:320(均质斑点)+1：160（肌动蛋白）；SMA+1：160，行肝脏穿刺活检，诊断为药物诱导的AIH（曾应用希刻劳等抗生素），服用醋酸泼尼松片（初为30mg/d，并逐渐规律减量）及硫唑嘌呤片（50mg/d）治疗，并于门诊规律复查调整用药剂量。3年前（2017年4月）因血清生化学指标恢复满意停用硫唑嘌呤与醋酸泼尼松龙，于2017年8月复查肝功能及IgG示ALT、AST、GGT、IgG再次升高，遂重新服用醋酸泼尼松片（初为30mg/d，并逐渐规律减量）及硫唑嘌呤片（50mg/d）治疗，规律复查肝功能及免疫球蛋白，指标恢复良好，激素逐渐减量，2月前停用激素，继续服用硫唑嘌呤至今。2年前复查时发现肿瘤标记物异常（CA199 44.3U/mL，CA-50 37.56 U/mL），行胃镜、结肠镜检查示胃息肉（山田I型），结肠多发息肉（山田I-IV型），并行EMR术。术后无腹痛、便血，无发热，无心悸、头晕、心慌等不适。1年前复查胃镜、结肠镜，行胃肠息肉钳除术并留取病理，病理结果提示：胃底腺胃粘膜组织，呈活动性慢性炎，伴小凹上皮增生；幽门腺胃粘膜组织，呈轻度慢性萎缩性炎，伴轻度肠上皮化生；息肉状结肠粘膜组织呈慢性炎。近期患者仍有间断乏力、腹胀、反酸、纳差，无黄疸，无鼻衄及齿龈出血，无皮肤瘙痒及白陶土色变，无明显关节疼痛、口干眼干及脱发，现为复查及进一步诊治收入我科。患者自发病以来，精神状态可，食欲不佳，睡眠差，夜间尿频，每晚排尿约3-4次，大便不成形，每日排便约2-3次，体重下降10余斤。 既往史: 高血压病史12年，血压最高达180/100mmHg，初口服代文，后改为络活喜、奥美沙坦降压治疗，目前血压控制在130/70mmHg；室性早搏病史5年，口服倍他乐克治疗；胆囊结石、胆囊腺肌症病史30余年，腔隙性脑梗死病史12年，慢性萎缩性胃炎病史5年，荨麻疹病史3年，骨质疏松症、甲状腺结节、白内障、高尿酸血症、肺多发结节、主动脉粥样硬化2年；否认糖尿病、精神疾病史。否认肝炎史、结核史、疟疾史。否认输血史，结肠癌术后约19年，术后规律化疗；右踝骨折术后约11年。过敏史：无，输血史：无，预防接种史：无，传染病史：无。其他系统回顾无特殊。 个人史：出生并久居于本地，否认疫水、疫区接触史，否认其他放射性物质及毒物接触史。免疫接种史不详。否认吸烟史，否认饮酒史。否认去过武汉、否认接触过从武汉来的人。 家族史：父母已逝，2弟2妹，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。 检查分类: 超声 检查项目名称: 彩超 检查部位:病房甲状腺及颈部淋巴彩超 检查所见: 甲状腺前后径：右叶：2.3cm，峡部厚0.7cm，左叶：2.0cm。  甲状腺形态正常，包膜完整，光滑，腺体回声均匀，CDFI：腺体内未见异常血流信号。双叶可见多发结节，左叶大者大小约0.6x0.4cm，呈囊实性，边界清，规则，未见血流信号；右叶大者大小约0.9x0.4cm，呈囊实性，边界清，规则，周围可见血流信号。  双颈部未见异常肿大淋巴结。 检查结论: 甲状腺多发结节，TI-RADS 3 类 生化检查: 颜色-尿液、白细胞(镜检)-尿液：-、微量白蛋白(mAlb)-尿液：-g/L、病理管型类型-尿液、隐血(OB)-尿液：-、白细胞(镜检)-尿液：0/uL、蛋白质-尿液：-g/L、红细胞(镜检)-尿液、粘液丝计数(MUCS#)-尿液：0/uL、亚硝酸盐(NIT)-尿液：-、备注-尿液、细菌计数(BACT#)-尿液：0/uL、胆红素(BIL)-尿液：-umol/L、维生素C(VitC)-静脉血：0mmol/L、---------------尿液：-----------、酸碱度(pH)-尿液：6.0、管型-尿液、透明管型计数(HYAL#)-尿液：0/uL、非鳞状上皮细胞计数-尿液：0/uL、鳞状上皮细胞计数(SQEP#)-尿液：0/uL、葡萄糖(Glu)-尿液：-mmol/L、白细胞团(WBCC)-尿液：0/uL、比重(SG)-尿液：1.015、---------------尿液：----------、尿胆原(URO)-尿液：-umol/L、病理管型计数-尿液：0/uL、酵母菌(YEAST)-尿液：0/uL、镜检-尿液、红细胞(镜检)-尿液：0/uL、白细胞(镜检)-尿液、酮体(KET)-尿液：-mmol/L、未分类结晶计数(UNCX#)-尿液：1/uL
Question: 请全程使用中文进行回答。请使用你的已有知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。
Thought: 患者老年女性，慢性病程，主要临床表现为乏力、食欲减退及恶心，查肝生化示ALT、AST、GGT 明显升高，ALB降低，IgG升高，ANA、SMA阳性，于我院行肝脏穿刺等检查诊断“自身免疫性肝炎”诊断明确，规律应用激素+硫唑嘌呤治疗，目前激素已停用，硫唑嘌呤维持治疗。2年前发现肿瘤标记物异常，胃镜、结肠镜检查考虑“胃息肉（山田I型），结肠多发息肉（山田I-IV型）”。
Action: Finish[自身免疫性肝炎]


Relevant Context: 入院情况：患者青年女性，慢性病程；反复肝功异常14月余，以转氨酶和胆红素升高为主，ANA最高1:1000，IgG最高5480mg/L；既往应用可致肝损伤药物（中药、抗生素等），曾多次激素治疗后好转，减量（4mg/d）后肝功异常反复加重；腹部超声提示肝脏高回声结节，考虑血管瘤，胆囊炎，脾大，腹水。BMI 22.6kg/m2 主诉：间断皮肤、巩膜黄染14月余 现病史：患者14月余前受凉出现寒战、发热，体温最高39.8℃，夜间明显，无咳嗽、咳痰，无腹痛、腹胀，无便秘、腹泻，无呕血、黑便，无陶土样大便，就诊于当地诊所予“连花清瘟颗粒、感冒清热颗粒、感冒疏风丸、布洛芬”等药物（具体剂量不详），治疗6天后上述症状无缓解，并出现皮肤、巩膜黄染，伴乏力、腹胀、恶心和纳差，无呕吐、腹痛，无皮肤瘙痒、关节肿痛，无口苦、口干，当地诊所予“头孢呋辛、奥美拉唑”等药物静脉输液，治疗4天后上述症状无缓解。（2020-08-10）于宁夏医科大学总医院查肝功异常：ALP/GGT 273/411U/L, ALT/AST 474/339U/L, TB/DB 134/50umol/L, ALB 29.9g/L，TG 3.18mmol/L；凝血功能：PTA 66.4%，FIB 1.78g/L；嗜肝病毒和非嗜肝病毒感染阴性；腹部超声示：肝脏高回声结节，考虑血管瘤，胆囊炎，脾大，腹水；CT示：肝内小钙化灶；MRCP示：胆囊炎、腹水，Gllisoon鞘膜积液，肝左叶类圆形异常信号灶；考虑诊断“肝损害 药物性肝损害？自身免疫性肝炎”，予“头孢哌酮舒巴坦”抗感染，“舒肝宁、甘草酸单胺半胱氨酸注射液、熊去氧胆酸”保肝、退黄、利胆，补白蛋白、利尿等对症治疗，入院治疗8天后患者未再出现寒战、发热，复查肝功提示胆红素持续升高（最高TB/DB 283/194umol/L）,PTA最低59.06%，口服甲泼尼龙40mg1次后出现低体温（＜35℃），即停药，先后予3次“人工肝治疗+血浆置换术（血浆前予地塞米松5mg）”清除胆红素、炎症因子，后皮肤巩膜黄染较前减轻，复查肝功能好转出院：TB/DB 72/51umol/L, ALP/GGT 133/314U/L, ALT/AST 116/145U/L，凝血功能：PTA 113%。院外口服熊去氧胆酸250mg tid，共20天。13月余前患者自觉皮肤巩膜黄染较前加重，无发热、寒战等不适，再次（2020-09-30）于宁夏医科大学总医院查肝功：TB/DB 79/20umol/L,ALP/GGT 252/355U/L, ALT/AST 284/654U/L,ALB/GLB 33/30g/L, TG 3.16mmol/L；凝血功能：PTA 113%，FIB 2.412g/L；ANA：1:100；IgG 19.3g/L,IgM 0.22g/L；AMA、AMA-M2、ASAM、自身免疫性肝病抗体谱阴性；ANCA抗体谱阴性；复查MRCP：肝左内叶异常信号灶，胆囊炎，Gllisoon鞘少量积液，右肾小囊肿，双侧胸腔少量积液；2020-10-19行肝脏穿刺活检术，病理提示：小叶结构正常，肝细胞明显肿胀，肝细胞内胆汁淤积，可见肝细胞点灶状坏死，小叶内可见多量淋巴细胞、中性粒细胞浸润，汇管区及周围多量淋巴细胞及中性粒细胞浸润，未见纤维化，中度界板性炎，符合胆管梗阻性中度肝损害性病变；予“丁二磺酸腺苷蛋氨酸、舒肝宁、甘草酸单胺半胱氨酸注射液、熊去氧胆酸”保肝、退黄、利胆，联合甲泼尼龙（8mg/日，服用7天后改为16mg/日）治疗，乏力和腹胀明显减轻，其余症状同前；2020-10-22复查肝功：TB/DB 144/63umol/L,ALP/GGT 108/537U/L,ALT/AST 24/90U/L,ALB/GLB 29/31g/L, TG 3.62mmol/L；凝血功能：PTA 72.6%，FIB 1.541g/L；转氨酶较前下降，于2020-10-23出院。院外口服熊去氧胆酸250mg tid，甲泼尼龙16mg qd。
11月余前（2020-11-14）患者自行停用甲泼尼龙并服用中药（具体不详）治疗1月，期间（2020-12-14）查肝功：TB/DB 104/68umol/L,ALP/GGT 169/396U/L,ALT/AST 584/1063U/L,ALB/GLB 31.6/77g/L；凝血功能：PTA 72.8%，皮肤巩膜黄染未见缓解，其余症状同前。遂于（2020-12-21）空军军医大学第一附属医院住院查肝功：TB/DB 54/41umol/L,ALP/GGT 103/236 U/L, ALT/AST 134/484U/L,ALB/GLB 93/64g/L, TG 2.94mmol/L；凝血功能：PT 12.4s，APTT 35.5s，PTA 77.6%，FIB 2.412g/L；ANA：1:1000阳性；IgG 54.8g/L,IgM 0.247g/L；铁代谢：转铁蛋白 2.73g/L，铁蛋白 334ug/L；血常规：WBC 3.78×109/L，淋巴细胞绝对值 0.84×109/L，RBC 3.15×1012/L，Hb 108g/L，MCV 107.6fl,MCH 24.2pg，MCHC 319 g/L，PLT 187×109/L；铜蓝蛋白阴性；肝弹性：肝脏硬度10.6kPa，脂肪衰减245dB/m；于2020-12-29行骨穿病理：骨髓红系增生；骨髓增生正常，红细胞系统增生占明显优势，以中、晚红为主，粒系比例减低，分别未见异常，巨核细胞数量正常。基质水肿，纤维组织未见增生。可溶性CD25/IL-2R阴性；NK细胞毒性颗粒检测提示：CD3-CD56+NK细胞占淋巴细胞2.77%，颗粒酶B阳性率85.61%，NK细胞杀伤活性减低；免疫蛋白固定电泳：IgG，Lambda单克隆免疫球蛋白渗透区带深染；尿免疫固定电泳：Kappa单克隆免疫球蛋白单克隆疑似阳性。考虑诊断“肝损害，噬血细胞综合症不除外，肝血管瘤”，予降转氨酶、补蛋白、补充维生素、口服甲泼尼龙32mg/日等治疗，2021-01-15复查肝功：TB/DB 18/10umol/L,ALP/GGT 56/96U/L, ALT/AST 105/47U/L,ALB/GLB 42/39 g/L, TG 2.94mmol/L；凝血功能：PT 12.5s，APTT 30s，PTA 80.5%，FIB 1.6g/L；血常规：WBC 7.5×109/L，RBC 3.24×1012/L，Hb 113g/L，MCV 107.4fl，MCH 34.9pg，MCHC 325 g/L，PLT 224×109/L，皮肤巩膜黄染较前减轻后出院。院外甲泼尼龙32mg/日（减4mg/1周）、醋酸钙胶囊1粒/日、维生素A、D1粒/日和维生素E1粒 tid、叶酸1粒/日、甲钴胺1粒 tid，口服3月余。6月余前患者甲泼尼龙用量减至4mg/日，自觉皮肤巩膜黄染加重，复查（2021-4-25）肝功转氨酶升高（ALT/AST 1298/1296U/L）,至解放军第四军医大学唐都医院治疗，甲泼尼龙用量改为48mg/日（减8mg/周），其余治疗同前。5月余前患者至协和门诊复查，建议口服易善复1粒 bid、双环醇1粒 bid、甲泼尼龙逐渐减量（减4mg/2周 ），后每月复查胆红素正常,转氨酶轻度升高。1月余前患者甲泼尼龙用量减至4mg/日，自觉皮肤巩膜黄染加重，遂于（2021-09-23）吴忠市人民医院复查肝功：TB/DB 88/44umol/L,ALP/GGT 173/438U/L, ALT/AST 2330/2327U/L,ALB/GLB 43.8/36.4g/L，入院保肝治疗后无好转，甲泼尼龙4mg/日（未停）。9天前患者于宁夏医科大学总医院（2021-10-05）查肝功：TB/DB 151/92umol/L,ALP/GGT 120/438U/L, ALT/AST 84/90U/L,ALB/GLB 31/28g/L,；血常规：WBC 4.51×109/L，RBC 3.18×1012/L，Hb 108g/L，PLT 148×109/L，予保肝治疗（具体不详）。院后口服甲泼尼龙4mg/日、乳果糖15ml bid、丁二磺酸腺苷蛋氨酸1g bid，其余口服药同前，服用至今。2天前患者于我院门诊查血常规+C反应蛋白(2021-10-12):RBC 3.56×1012/L，MCV 107.10fL，PLT 198× 109/L，HGB 121g/L，WBC 4.93×109/L，为查明肝功异常原因，收治我院。自发病来，患者意识清，精神差，饮食差，睡眠欠佳，尿深茶色，大便无殊，近1年体重下降5kg。 既往史: 31年前曾患甲肝，已治愈。否认高血压、心脏病史，否认糖尿病、脑血管病、精神疾病史。否认结核史、疟疾史。否认手术史、传染病史。有输血史：2020年8月21日、25日和28日于宁夏医科大学总医院行人工肝治疗+血浆置换术（共3次）。有过敏史：海鲜过敏，否认药物过敏。预防接种史不详，其他系统回顾无特殊。 个人史：出生并久居于宁夏银川，否认疫水、疫区接触史，否认其他放射性物质及毒物接触史。免疫接种史不详。否认吸烟史，否认饮酒史。否认去过疫区、否认接触过从疫区来的人。 家族史：父母健在，1弟，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。 检查分类: MRI 检查项目名称: 上腹部 平扫+增强 检查部位:上腹部 检查所见: 肝脏外形光滑，各叶比例适中。肝S4见类圆形T1WI低信号影，T2WI呈高信号，增强扫描动脉期边缘结节状强化，门脉期及延迟期呈可见造影剂向中心填充，病变大小约为3.8cm×2.7cm。肝内见少许小点状水样信号影，边界清楚，增强扫描未见强化。肝内外胆管无扩张，胆囊不大，壁不厚，腔内未见异常信号影。      脾不大，实质信号未见异常。胰腺形态如常，实质信号未见异常，胰管无扩张。      双肾形态如常，其内均可见圆形水样信号影，较大者位于右肾，大小约为0.6cm×0.6cm，未见强化。双肾盂及输尿管上段未见扩张，腔内未见异常信号。双侧肾上腺形态、信号未见异常。      腹腔及腹膜后间隙未见肿大淋巴结。 检查结论: 1、肝S4血管瘤；         肝内少许小囊肿；      2、双肾多发囊肿。 生化检查: 备注-静脉血(空腹)、标本质量-静脉血(空腹)、铜蓝蛋白(CER)-静脉血(空腹)：0.18g/L
Question: 请全程使用中文进行回答。请使用你的已有知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。
Thought: 患者青年女性，慢性病程；因发热后出现反复肝功异常14月余，以转氨酶和胆红素升高为主，既往ANA：1:1000阳性；IgG 54.8g/L，腹部超声示：肝脏高回声结节，考虑血管瘤，胆囊炎，脾大，腹水，间断皮肤、巩膜黄染，伴乏力。既往有中药、抗生素和激素用药史，曾多次用激素减量后病情反复。考虑“肝功能异常原因待查”收治入院，需完善相关检查进一步鉴别。
Action: Finish[肝血管瘤+自身免疫性肝炎+药物性肝损伤]

"""

#################################
'''
借助GPT4o，生成上述COT的反思过程，从而生成REFLECT链条。
GPT基于的指示如下：
请帮忙生成上述fewshot的反思文本，反思Relevant Context是如何通过Thought进行信息提取和学习，并且最终输出正确答案Action
'''
# 样本来自于context字数最少的96,191
# TODO: 更加灵活化，减少token
# TODO: gpt4o给出的反思也很大可能是胡说八道


MED_COT_REFLECT="""
Relevant Context: 入院情况：1.老年男性，慢性病程；2.半年前无明显诱因出现上腹部隐痛不适，阵发性，可耐受，时有胀痛，有恶心，未吐，伴有大便习惯及性状改变，表现为3-7天排便一次，时有腹泻、便秘交替进行，无粘液脓血便，无呕血、黑便，无发热，曾行腹部超声检查提示脂肪肝，胆囊腺肌症，为求进一步治疗，收入我科。 主诉：间断上腹部疼痛半年。 现病史：患者半年前无明显诱因出现上腹部隐痛不适，阵发性，可耐受，时有胀痛，有恶心，未吐，伴有大便习惯及性状改变，表现为3-7天排便一次，时有腹泻、便秘交替进行，无粘液脓血便，无呕血、黑便，无发热，曾行腹部超声检查提示脂肪肝，胆囊腺肌症，为求进一步治疗，收入我科。病来精神状态可，无消瘦、乏力，无排尿正常，体重无增减。 既往史: 否认高血压、心脏病史，否认糖尿病、脑血管病、精神疾病史。30余年前肺结核病史，已治愈。否认肝炎史、疟疾史。手术史：有，于2018年04月28日因中耳炎全麻下行左侧完壁式乳突根治+后鼓室开放+鼓室成型术（II型）。过敏史：无，输血史：无，预防接种史：无，传染病史：无。其他系统回顾无特殊。 个人史：出生原籍，久居于本地，否认疫水、疫区接触史，否认其他放射性物质及毒物接触史。免疫接种史不详。否认吸烟史，有饮酒史，否认去过疫区、否认接触过从疫区来的人。饮酒史40余年，42-56°白酒3两/天；未戒酒。 家族史：父母已逝，兄弟姐妹9人，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。 检查分类: 超声 检查项目名称: 病房血管彩超 检查部位:病房颈动脉彩超 检查所见: 颈动脉:  双侧颈总动脉、颈内动脉、颈外动脉可视段管径对称，内-中膜不均匀增厚，左侧颈总动脉分叉处后壁见大小约13.9x4.0mm等回声斑块，管腔未见明显狭窄和局限性扩张，管腔内可见血流充盈，斑块处血流充盈缺损，血流速度正常。 检查结论: 双侧颈动脉内-中膜不均匀增厚伴左侧斑块形成 生化检查: 标本质量-静脉血(空腹)、总前列腺特异性抗原(TPSA)-静脉血(空腹)：4.43ng/mL、备注-静脉血(空腹)、游离前列腺特异性抗原(FPSA)-静脉血(空腹)：0.73ng/mL
Question: 请全程使用中文进行回答。请使用你的已有知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。
Thought: 患者老年男性，慢性病程；长期大量饮酒史，超声提示脂肪肝。
Action: Finish[酒精性肝病]
Reflection: 在分析患者的病史时，几个关键点引起了注意：慢性病程和长期饮酒史：患者有40余年的饮酒史，且每天饮用42-56°白酒3两，表明其肝脏可能长期受酒精的影响。脂肪肝的超声检查提示：超声检查已明确显示患者患有脂肪肝，这是酒精性肝病的常见表现。未提及药物使用史或自身免疫相关症状：患者否认药物使用史，也无自身免疫性肝炎的典型症状，如黄疸或肝功能异常。根据这些信息，可以推断患者最有可能患有与长期饮酒相关的酒精性肝病，同时存在脂肪肝。未发现明确的证据支持药物性肝损伤、原发性肝癌、自身免疫性肝炎、肝囊肿或肝血管瘤的诊断。


Relevant Context: 入院情况：患者主因“发现肝功异常3年”入院。患者肝功异常主要以ALT、GGT升高为主，保肝治疗GGT呈下降趋势，后无明显诱因逐渐升高。无明显不适症状，平素接触有机物（防护可）。常见铜铁代谢未见异常，抗核抗体谱未见明显异常。 主诉：发现肝功异常3年。 现病史：患者3年前自诉熬夜后体检发现肝功异常：谷丙转氨酶64u/l（未见具体报告）。无食欲下降，无乏力，无腹痛，无尿黄，未予重视。2年前复查肝功仍异常：谷丙转氨酶及谷氨酰转肽酶较前增高（未见报告），于中日友好医院就诊，行甲肝、乙肝、丙肝、戊肝检查未提示异常，诊断“肝功能异常”，给予水飞蓟素治疗。自10月前（2020年8月）起后每月复查一次肝功：ALT波动于30-70u/l。GGT波动于120-200u/l。6月前加用熊去氧胆酸治疗，GGT呈下降趋势，近3月逐渐升高，1月前升高至450u/l，ALT升高至123u/l。2周前于我院门诊就诊，行检查：免疫球蛋白:IgG 1410.0*mg/dl，IgA 285.0*mg/dl，IgM 52.5*mg/dl，铜兰蛋白(CER)0.21*g/l，铁蛋白(Ferritin)164.00*ng/ml，抗核抗体谱:间接免疫荧光法抗核抗体 ()阴性*，间接免疫荧光法双链DNA(dsDNA)阴性*。为进一步诊治收入我科。患者自发病来，神清，精神可，大小便未见明显异常，体重增加8Kg。 既往史: 过敏性鼻炎11年，未予规律诊治。否认高血压、心脏病史，否认糖尿病、脑血管病、精神疾病史。否认肝炎史、结核史、疟疾史。手术史：无，过敏史：有，青霉素类药物过敏。输血史：无，预防接种史：无，传染病史：无。其他系统回顾无特殊。 个人史：出生并久居于本地，否认疫水、疫区接触史，否认其他放射性物质。近3年接触有机物：异丙醇、乙醇、钨类物质。免疫接种史不详。否认吸烟史，否认饮酒史。否认去过疫区、否认接触过从疫区来的人。 家族史：父母健在，否认家族中类似病史、传染病史、遗传病史及肿瘤病史。 检查分类: 超声 检查项目名称: nan 检查部位:nan 检查所见: nan 检查结论: nan 生化检查: 备注-静脉血、标本质量-静脉血、CMV病毒拷贝数-静脉血：<250copies/mL、CMV病毒核酸检测-静脉血：2205
Question: 请全程使用中文进行回答。请使用你的已有知识，判断该病人是以下这7个病中的哪一种或多种疾病。脂肪肝，药物性肝损伤，原发性肝癌，酒精性肝病，自身免疫性肝炎，肝囊肿，肝血管瘤。如果病人是多种疾病，用'+'号连接。
Thought: 患者青年男性，慢性病程；3年前自诉熬夜后体检发现肝功异常：谷丙转氨酶稍高于正常。无食欲下降，无乏力，无腹痛，无尿黄。2年前外院诊断“肝功能异常”，给予水飞蓟素治疗。10月前起后每月复查一次肝功：ALT波动于30-70u/l，GGT波动于120-200u/l。半年前给予熊去氧胆酸治疗，GGT下降，近3月升高，1月前GGT升高至450u/l，ALT升高至123u/l。2周前于我院门诊就诊，行检查：铜铁代谢未见异常，抗核抗体未见异常，停药。结合患者接触史及辅助检查，暂考虑该诊断。
Action: Finish[脂肪肝]
Reflection: 根据患者的病史，肝功能异常持续存在，且与药物或自身免疫因素无关的可能性较大。考虑到患者有较长时间的有机物接触史，且经过治疗后GGT和ALT反复波动，说明患者可能存在脂肪肝或与其职业接触相关的肝损伤。没有证据支持药物性肝损伤、自身免疫性肝炎或酒精性肝病的诊断，也没有检查结果提示存在肝囊肿或肝血管瘤。基于上述分析，正确的诊断是Finish[脂肪肝]。这一结论是根据患者的接触史和辅助检查结果，排除了其他可能的诊断，最终选择了脂肪肝作为最可能的疾病。

"""