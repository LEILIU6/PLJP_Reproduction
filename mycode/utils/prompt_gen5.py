from ..utils import loader
import json
import re


article_prompt = """
你是一个法律法条预测的专家，擅长进行法律法条预测。

案情描述包括三个主要描述：主观动机、客观行为和事外情节。你将会收到一个#待预测的案情描述，以及3个类似的案例。请根据待预测案情描述与案例案情描述的相关性，从3个候选法条中选出与本案最相关法条。

# 待预测的案情描述
{fact}

# 案例+候选法条
{examples}

# 判断逻辑
1. 先分析待预测的案情属于哪种罪名，并给出分析
2. 再根据案情描述的相似度，找出与本案最相关的案例,并给出分析；
3. 选择最相关案例对应的法条作为本案的相关法条，相关法条有且只有1个；

# 输出格式
{{
    "step1待预测的案情具体分析": "xxxxxx",
    "step2与本案最相关的案例具体分析": "xxxxxx",
    "相关法条": "xxx"
}}
"""

charge_prompt = """
你是一个罪名判定的专家，擅长进行案件的罪名判定。

案情描述包括三个主要描述：主观动机、客观行为和事外情节。你将会收到一个待判定的案情描述，以及3个类似的案例。请根据待判定的案情描述与案例案情描述的相关性，从3个候选罪名中选出与本案最相关的罪名。

# 本案案情
{fact}
本案的相关法条是：{fact_article}

# 案例
## 案例1
{example_1}

## 案例2
{example_2}

## 案例3
{example_3}

# 判断逻辑
1. 理解待预测案情和3个案例案情；
2. 根据案情描述的相似度，找出与本案最相关的案例；
3. 选择最相关案例对应的罪名作为本案的判定罪名，判定罪名有且只有1个；

# 输出格式
1. 先根据判断逻辑一步一步地分析，陈述你的理由；
2. 最后输出你的答案，答案使用“$$”符号修饰，比如: $$故意毁坏财物$$, $$重大责任事故$$。

现在，请开始你的分析：
"""

pt_cls2str = ["其他", "六个月以下", "六到九个月", "九个月到一年", "一到两年", "二到三年", "三到五年", "五到七年", "七到十年", "十年以上"]

def get_pt_cls(pt):
    if pt > 10 * 12:
        pt_cls = 9
    elif pt > 7 * 12:
        pt_cls = 8
    elif pt > 5 * 12:
        pt_cls = 7
    elif pt > 3 * 12:
        pt_cls = 6
    elif pt > 2 * 12:
        pt_cls = 5
    elif pt > 1 * 12:
        pt_cls = 4
    elif pt > 9:
        pt_cls = 3
    elif pt > 6:
        pt_cls = 2
    elif pt > 0:
        pt_cls = 1
    else:
        pt_cls = 0
    return pt_cls


def label_prompt_case(case, task):
    if task == "charge":
        label = case["meta"]["accusation"][0]
    if task == "article":
        label = case["meta"]["relevant_articles"][0]
        label = f"第{label}条"
    if task == "penalty":
        pt = case["meta"]["term_of_imprisonment"]["imprisonment"]
        pt_cls = get_pt_cls(pt)
        label = pt_cls2str[pt_cls] + "有期徒刑"
    return label

def label_prompt(label, task):
    if task == "charge":
        label = label
    if task == "article":
        label = f"第{label}条"
    if task == "penalty":
        label = label
    return label

def retrieved_label_option_fewshot(fact, in_context_content, args, caseID):

    if args.task in ["charge", "penalty"]:
        predicted_articles = {}
        with open(args.predicted_article_path, encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                case = json.loads(line)
                _article = re.findall(r'\$\$(.*?)\$\$', case["choices"][0]["message"]["content"])[-1]
                _article = re.findall(r'\d+', _article)[0]
                predicted_articles[case["caseID"]] = _article
        predicted_article = predicted_articles[caseID]
        predicted_article_content = loader.truncate_text(loader.load_predicted_article_content(predicted_article), max_len=100) 
    
    if args.task in["penalty"]:
        predicted_charges = {}
        with open(args.predicted_charge_path, encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                case = json.loads(line)
                _charge = re.findall(r'\$\$(.*?)\$\$', case["choices"][0]["message"]["content"])[-1]
                predicted_charges[case["caseID"]] = _charge  
        predicted_charge = predicted_charges[caseID]


    task_prompt = {
        "charge": "本案的被告人罪名是",
        "article": "本案的相关法条是",
        "penalty": "本案的被告人刑期是",
    }
    examples = in_context_content["precedents"]
    # 0shot
    # examples = []
    label_option = in_context_content["topk_label_option"]

    texts = []
    for case in examples:
        cur_fact = case["fact"].replace(" ", "")
        if args.use_split_fact: # reframed fact
            cur_fact = [case["fact_split"]["zhuguan"], case["fact_split"]["keguan"], case["fact_split"]["shiwai"]]
            cur_fact = "。".join(cur_fact)
        
        article_label = label_prompt_case(case, "article")
        charge_label = label_prompt_case(case, "charge")
        penalty_label = label_prompt_case(case, "penalty")

        max_examplar_len = 256
        cur_fact = loader.truncate_text(cur_fact, max_len = max_examplar_len)
        if args.task == "article":
            text = f"{cur_fact}\n{task_prompt['article']}：{article_label}"
        elif args.task == "charge":
            text = f"{cur_fact}\n{task_prompt['article']}：{article_label}\n{task_prompt['charge']}：{charge_label}"
        elif args.task == "penalty":
            text = f"{cur_fact}\n{task_prompt['article']}：{article_label}\n{task_prompt['charge']}：{charge_label}\n{task_prompt['penalty']}：{penalty_label}"
        
        texts.append(text)

    label_option = [label_prompt(i, args.task) for i in label_option[:args.shot]]   # topk-option
    tok_label = "；".join(label_option)
    #0 shot  
    # tok_label = []
    texts = texts[:args.shot]  # few-shot

    if args.task == "article": # 加入article definition
        article_definition = in_context_content["article_definition"]
        article_definition = [loader.truncate_text(text, max_len=50) for text in article_definition]
        article_definition = article_definition[:len(texts)]
        examples_str = ""
        for i in range(len(texts)):
            examples_str += f"## 案例{i+1}\n{texts[i]}\n{article_definition[i]}\n\n"

    #法条
    if args.task == "article":
        return article_prompt.format(fact=fact, 
                                     examples=examples_str)

    #罪名
    elif args.task == "charge":
        return charge_prompt.format(fact=fact,
                                    fact_article=predicted_article_content,
                                    example_1=texts[0],
                                    example_2=texts[1],
                                    example_3=texts[2])
    #刑期
    elif args.task == "penalty":
        return f"用<>括起来的是本案事实，用---括起来的是与本案内容类似的"+str(args.shot)+f"个案件，用```括起来的是本案的相关法条,法条内容和相关罪名。请通过主观动机，客观行为与事外情节三个方面,理解与比较用---括起来的"+str(args.shot)+f"个案件与本案事实的异同，并参考本案的相关法条、法条内容和相关罪名，选择最终的刑期。注意：请输出本案的刑期，并结合类案给出选择本案刑期的理由。\n"+f"<"+fact+f">"+f"\n---"+examples +f"---\n"+f"```"+predicted_article+f" "+predicted_article_content+f" "+predicted_charge+f"```"+f"\n{task_prompt[args.task]}以下几个选项的其中之一：[{tok_label}]" + f"\n{task_prompt[args.task]}："

    
def fact_split(fact):
    prompt = f"""
一段法院查明的犯罪事实可以区分成：主观动机、客观行为以及事外情节。
其中，主观动机是指行为人对自己实施的危害社会的行为及其结果所持的心理态度，包括犯罪的故意、过失，犯罪的动机、目的等。
客观行为是指构成犯罪在客观活动方面所必须具备的条件，包括危害行为、危害结果，以及危害行为与危害结果之间的因果关系等。
事外情节是指决定刑罚轻重时根据的各种事实情况，从轻处罚的情节包括自首、有立功表现等，从重处罚的情节包括累犯等。
下面提供两个参考示例。根据以上信息，你的任务是将用```括起来的犯罪事实进行归纳，并总结成与参考示例同样的格式。
字数限制在200字以内。

示例:
事实：经审理查明，被告人徐佑华舞厅老板，携带事先在农药店买好的农药，来到舞厅，将农药随机投放在舞厅进门处座位上的两杯茶水里。经鉴定，两杯茶水中均检出农药毒死蜱成分。被告人徐佑华被公安机关抓获归案。但被告人徐佑华归案后如实供述其罪行，依法可以从轻处罚。
主观动机：被告人徐佑华在公共场所故意投放毒害性物质
客观行为：被告人徐佑华舞厅老板，携带事先在农药店买好的农药，来到舞厅，将农药随机投放在舞厅进门处座位上的两杯茶水里。经鉴定，两杯茶水中均检出农药毒死蜱成分。
事外情节：但被告人徐佑华归案后如实供述其罪行，依法可以从轻处罚。

```{fact}```
"""
    return prompt
