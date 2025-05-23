import os
import json
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
print(sys.path)


def process(raw_text):
    system_prompt = raw_text.split('[BEGIN DATA]')[0].strip()

    question_match = re.search(r"\[Query\]:(.*?)\[Response 1\]", raw_text, re.DOTALL)
    response1_match = re.search(r"\[Response 1\]:(.*?)\[Response 2\]", raw_text, re.DOTALL)
    response2_match = re.search(r"\[Response 2\]:(.*?)\[END DATA\]", raw_text, re.DOTALL)
    criteria_match  = raw_text.split('[END DATA]')[-1].strip()

    # 去掉换行和前后空格
    question = question_match.group(1).strip() if question_match else ''
    response1 = re.sub(r'^\*{3,}\s*', '', response1_match.group(1).strip()) if response1_match else ''
    response2 = response2_match.group(1).strip() if response2_match else ''
    clean_text = lambda x: re.sub(r'^\*{3,}\s*|\s*\*{3,}$', '', x.strip(), flags=re.MULTILINE)

    question = clean_text(question_match.group(1)) if question_match else ''
    response1 = clean_text(response1_match.group(1)) if response1_match else ''
    response2 = clean_text(response2_match.group(1)) if response2_match else ''
    return system_prompt, question, response1, response2, criteria_match

def generate_by_gpt(model_name, dataset_path, output_path):
    # import sys
    # from pathlib import Path
    # sys.path.append(str(Path(__file__).resolve().parent.parent))
    # print(sys.path)
    # os.environ["OPENAI_API_KEY"] = ''

    from judge_utils.litellm import LiteLLM, AsyncLiteLLM
    from judge_utils.judge import AGDEval
    from judge_utils.prompts import RELATIVE_PROMPT, RELATIVE_PROMPT_v2,PAIREWISE_AUTOJ 
    model = AsyncLiteLLM(model_name, api_base='https://api.openai.com/v1', requests_per_minute=100) 

    judge = AGDEval(model=model, relative_grade_template=PAIREWISE_AUTOJ)


    dataset = json.load(open(dataset_path))
    print(len(dataset))
    instructions =[]
    responses_A = []
    responses_B = []

    for i in range(len(dataset)):
        data = dataset[i]
        system_prompt, question, response1, response2, criteria_match = process(data["usrmsg"])
        instructions.append(question)
        responses_A.append(response1)
        responses_B.append(response2)


    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        mode = "Auto-J",
        model_name="",
    )
    data_list = []
    for idx,(feedback, score) in enumerate(zip(feedbacks, scores)):
        if score == 'A': 
            score = 0
        elif score == 'B':
            score = 1
        elif score == 'TIE':
            score = 2
        else:
            score = -1
        data_list.append({
            "id": idx,
            "usrmsg": dataset[idx]["usrmsg"],
            "target_output": feedback,
            "gt_label": dataset[idx]["gt_label"],
            "pred_label": score,
            "instruction": instructions[idx],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)


def generate_by_llm(model_path, dataset_path, output_path, model_name):
    from judge_utils.vllm import VLLM
    from judge_utils.judge import AGDEval
    from judge_utils.prompts import PAIREWISE_AUTOJ


    model = VLLM(model = model_path, lora_config=None, max_model_len = 4096, gpu_memory_utilization = 0.8)

    judge = AGDEval(model=model, relative_grade_template=PAIREWISE_AUTOJ)

    dataset = json.load(open(dataset_path))
    print(len(dataset))
    instructions =[]
    responses_A = []
    responses_B = []

    for i in range(len(dataset)):
        data = dataset[i]
        system_prompt, question, response1, response2, criteria_match = process(data["usrmsg"])
        instructions.append(question)
        responses_A.append(response1)
        responses_B.append(response2)

    params = {
        "max_tokens": 1024,
        "repetition_penalty": 1.03,
        "best_of": 1,
        "temperature": 1.0,
        "top_p": 0.9,
        "seed": 42,
    }

    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        params=params,
        mode = "Auto-J",
        model_name = model_name,
    )

    data_list = []
    for idx,(feedback, score) in enumerate(zip(feedbacks, scores)):
        if score == 'A': 
            score = 0
        elif score == 'B':
            score = 1
        elif score == 'TIE':
            score = 2
        else:
            score = -1
        data_list.append({
            "id": idx,
            "usrmsg": dataset[idx]["usrmsg"],
            "target_output": feedback,
            "gt_label": dataset[idx]["gt_label"],
            "pred_label": score,
            "instruction": instructions[idx],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)



def generate_critique_by_gpt(model_name, dataset_path, output_path):
    # os.environ["OPENAI_API_KEY"] = ''
    from judge_utils.litellm import LiteLLM, AsyncLiteLLM
    from judge_utils.judge import AGDEval
    from judge_utils.prompts import CRITICIZE
    model = AsyncLiteLLM(model_name, api_base='https://api.openai.com/v1', requests_per_minute=100) 
    judge = AGDEval(model=model, relative_grade_template=CRITICIZE)


    dataset = json.load(open(dataset_path))
    # dataset = dataset[:10] ###
    print(len(dataset))
    instructions =[]
    responses_A = []
    responses_B = []

    for i in range(len(dataset)):
        data = dataset[i]
        system_prompt, question, response1, response2, criteria_match = process(data["usrmsg"])
        instructions.append(question)
        responses_A.append(response1)
        responses_B.append(response2)


    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        mode = "no_parse",
        model_name="",
    )
    data_list = []
    for idx,(feedback, score) in enumerate(zip(feedbacks, scores)):
        data_list.append({
            **dataset[idx],
            "critique": feedback,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

def generate_critique_by_llm(model_path, dataset_path, output_path, model_name):
    from judge_utils.vllm import VLLM
    from judge_utils.judge import AGDEval
    from judge_utils.prompts import CRITICIZE


    model = VLLM(model = model_path, lora_config=None, max_model_len = 4096, gpu_memory_utilization = 0.8)

    judge = AGDEval(model=model, relative_grade_template=CRITICIZE)

    dataset = json.load(open(dataset_path))
    print(len(dataset))
    instructions =[]
    responses_A = []
    responses_B = []

    for i in range(len(dataset)):
        data = dataset[i]
        system_prompt, question, response1, response2, criteria_match = process(data["usrmsg"])
        instructions.append(question)
        responses_A.append(response1)
        responses_B.append(response2)

    params = {
        "max_tokens": 1024,
        "repetition_penalty": 1.03,
        "best_of": 1,
        "temperature": 1.0,
        "top_p": 0.9,
        "seed": 42,
    }

    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        params=params,
        mode = "no_parse",
        model_name = model_name,
    )

    data_list = []
    for idx,(feedback, score) in enumerate(zip(feedbacks, scores)):
        data_list.append({
            **dataset[idx],
            "critique": feedback,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

def generate_final_feedback_by_gpt(model_name, dataset_path, output_path):
    # os.environ["OPENAI_API_KEY"] = ''
    from judge_utils.litellm import LiteLLM, AsyncLiteLLM
    from judge_utils.judge import AGDEval
    from judge_utils.prompts import GENERATE_NEW_FEEDBACK_SINGLE
    model = AsyncLiteLLM(model_name, api_base='https://api.openai.com/v1', requests_per_minute=100) 
    judge = AGDEval(model=model, relative_grade_template=GENERATE_NEW_FEEDBACK_SINGLE)


    dataset = json.load(open(dataset_path))
    dataset = dataset[:10] ###
    print(len(dataset))
    instructions =[]
    responses_A = []
    responses_B = []

    for i in range(len(dataset)):
        instructions.append(dataset[i]["usrmsg"])
        responses_A.append(dataset[i]["feedback(teacher)"])
        responses_B.append(dataset[i]["critique"])


    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        mode = "Auto-J",
        model_name="",
    )
    data_list = []
    for idx,(feedback, score) in enumerate(zip(feedbacks, scores)):
        data_list.append({
            **dataset[idx],
            "new_feedback": feedback,
            "new_label": score,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

def generate_final_feedback_by_llm(model_path, dataset_path, output_path, model_name):
    from judge_utils.vllm import VLLM
    from judge_utils.judge import AGDEval
    from judge_utils.prompts import GENERATE_NEW_FEEDBACK_SINGLE


    model = VLLM(model = model_path, lora_config=None, max_model_len = 4096, gpu_memory_utilization = 0.8)

    judge = AGDEval(model=model, relative_grade_template=GENERATE_NEW_FEEDBACK_SINGLE)

    dataset = json.load(open(dataset_path))
    print(len(dataset))
    instructions =[]
    responses_A = []
    responses_B = []

    for i in range(len(dataset)):
        instructions.append(dataset[i]["usrmsg"])
        responses_A.append(dataset[i]["feedback(teacher)"])
        responses_B.append(dataset[i]["critique"])

    params = {
        "max_tokens": 1024,
        "repetition_penalty": 1.03,
        "best_of": 1,
        "temperature": 1.0,
        "top_p": 0.9,
        "seed": 42,
    }

    feedbacks, scores = judge.relative_grade(
        instructions=instructions,
        responses_A=responses_A,
        responses_B=responses_B,
        params=params,
        mode = "Auto-J",
        model_name = model_name,
    )

    data_list = []
    for idx,(feedback, score) in enumerate(zip(feedbacks, scores)):
        data_list.append({
            **dataset[idx],
            "new_feedback": feedback,
            "new_label": score,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)


def filter_raw(dataset_path, output_path):
    dataset = json.load(open(dataset_path))
    new_dataset = []
    for i, data in enumerate(dataset):
        if data["pred_label"] == -1 or data["pred_label"] == 2:
            continue
        else:
            assert data["pred_label"] == 0 or data["pred_label"] == 1
            new_dataset.append(data)
    print(len(new_dataset))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)



def filter_margin_for_refine(teacher_dataset_raw, assistant_dataset_raw, teacher_dataset_margin, assistant_dataset_margin, threshold, output_path):

    dataset_gpt4_raw = json.load(open(teacher_dataset_raw))
    dataset_gpt3_5_raw  = json.load(open(assistant_dataset_raw))

    dataset_gpt4 = json.load(open(teacher_dataset_margin))
    dataset_gpt3_5 = json.load(open(assistant_dataset_margin))
    dataset = []

    l = 0
    r = 0
    idx = 0
    print(f"Raw training dataset size: {len(dataset_gpt3_5_raw)}")
    if len(dataset_gpt3_5_raw) != len(dataset_gpt4_raw):
        raise ValueError("The lengths of the datasets are not equal.")

    labels = []
    for i in range(len(dataset_gpt3_5_raw)):
        if dataset_gpt3_5_raw[i]["pred_label"] == -1 or dataset_gpt3_5_raw[i]["pred_label"] == 2:
            if dataset_gpt4_raw[i]["pred_label"] == 2:
                labels.append({
                    "gpt3.5": -1,
                    "gpt4": -1,
                })
            else:
                assert dataset_gpt4_raw[i]["usrmsg"] == dataset_gpt4[r]["usrmsg"]
                labels.append({
                    "gpt3.5": -1,
                    "gpt4": int(dataset_gpt4[r]["margin"] > threshold),
                })
                r += 1
        else:
            assert dataset_gpt3_5_raw[i]["usrmsg"] == dataset_gpt3_5[l]["usrmsg"]
            if dataset_gpt4_raw[i]["pred_label"] == 2:
                labels.append({
                    "gpt3.5": int(dataset_gpt3_5[l]["margin"] > threshold),
                    "gpt4": -1,
                })
                l += 1
            else:
                assert dataset_gpt4_raw[i]["usrmsg"] == dataset_gpt4[r]["usrmsg"]
                labels.append({
                    "gpt3.5": int(dataset_gpt3_5[l]["margin"] > threshold),
                    "gpt4": int(dataset_gpt4[r]["margin"] > threshold),
                })
                
                if dataset_gpt3_5[l]["margin"] > threshold and dataset_gpt4[r]["margin"] > threshold:
                    assert dataset_gpt3_5[l]["pred_label"] == dataset_gpt4[r]["pred_label"]
                    dataset.append({
                        "idx": idx,
                        "usrmsg": dataset_gpt3_5[l]["usrmsg"],
                        "feedback(assistant)": dataset_gpt3_5[l]["target_output"],
                        "feedback(teacher)": dataset_gpt4[r]["target_output"],
                        "gt_label": dataset_gpt3_5[l]["gt_label"],
                        "pred_label": dataset_gpt3_5[l]["pred_label"],
                        "margin": dataset_gpt3_5[l]["margin"]
                    })
                    idx += 1
                l += 1
                r += 1
        

    data_dict = {}
    for i in range(len(labels)):
        label_str = str(labels[i]["gpt3.5"]) + "_" + str(labels[i]["gpt4"])
        data_dict[label_str] = data_dict.get(label_str, 0) + 1

    print(data_dict)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
def merge_and_tranfrom(teacher_dataset_raw, assistant_dataset_raw, teacher_dataset_margin, assistant_dataset_margin, threshold, add_dataset_path, output_path):

    dataset_gpt4_raw = json.load(open(teacher_dataset_raw))
    dataset_gpt3_5_raw = json.load(open(assistant_dataset_raw))

    dataset_gpt4 = json.load(open(teacher_dataset_margin))
    dataset_gpt3_5 = json.load(open(assistant_dataset_margin))
    
    add_dataset = json.load(open(add_dataset_path))
    new_dataset_path = output_path
    new_dataset = []


    l = 0
    r = 0
    idx = 0
    print(f"Dataset size: {len(dataset_gpt3_5_raw)}")
    if len(dataset_gpt3_5_raw) != len(dataset_gpt4_raw):
        print("Error: The lengths of the datasets are not equal.")

    labels = []
    for i in range(len(dataset_gpt3_5_raw)):
        if dataset_gpt3_5_raw[i]["pred_label"] == -1 or dataset_gpt3_5_raw[i]["pred_label"] == 2:
            if dataset_gpt4_raw[i]["pred_label"] == 2:
                labels.append({
                    "gpt3.5": -1,
                    "gpt4": -1,
                })
            else:
                assert dataset_gpt4_raw[i]["usrmsg"] == dataset_gpt4[r]["usrmsg"]
                labels.append({
                    "gpt3.5": -1,
                    "gpt4": int(dataset_gpt4[r]["margin"] > threshold),
                })
                if dataset_gpt4[r]["margin"] > threshold:
                    new_dataset.append({
                        "idx": idx,
                        "usrmsg": dataset_gpt4[r]["usrmsg"],
                        "target_output": dataset_gpt4[r]["target_output"],
                        "gt_label": dataset_gpt4[r]["gt_label"],
                        "pred_label": dataset_gpt4[r]["pred_label"],
                        "margin": dataset_gpt4[r]["margin"]
                    })
                    idx += 1
                r += 1
        else:
            assert dataset_gpt3_5_raw[i]["usrmsg"] == dataset_gpt3_5[l]["usrmsg"]
            if dataset_gpt4_raw[i]["pred_label"] == 2:
                labels.append({
                    "gpt3.5": int(dataset_gpt3_5[l]["margin"] > threshold),
                    "gpt4": -1,
                })
                if dataset_gpt3_5[l]["margin"] > threshold:
                    new_dataset.append({
                        "idx": idx,
                        "usrmsg": dataset_gpt3_5[l]["usrmsg"],
                        "target_output": dataset_gpt3_5[l]["target_output"],
                        "gt_label": dataset_gpt3_5[l]["gt_label"],
                        "pred_label": dataset_gpt3_5[l]["pred_label"],
                        "margin": dataset_gpt3_5[l]["margin"]
                    })
                    idx += 1
                l += 1
            else:
                assert dataset_gpt4_raw[i]["usrmsg"] == dataset_gpt4[r]["usrmsg"]
                labels.append({
                    "gpt3.5": int(dataset_gpt3_5[l]["margin"] > threshold),
                    "gpt4": int(dataset_gpt4[r]["margin"] > threshold),
                })
                if dataset_gpt4[r]["margin"] > threshold and dataset_gpt3_5[l]["margin"] <= threshold:
                    new_dataset.append({
                        "idx": idx,
                        "usrmsg": dataset_gpt4[r]["usrmsg"],
                        "target_output": dataset_gpt4[r]["target_output"],
                        "gt_label": dataset_gpt4[r]["gt_label"],
                        "pred_label": dataset_gpt4[r]["pred_label"],
                        "margin": dataset_gpt4[r]["margin"]
                    })
                    idx += 1
                if dataset_gpt3_5[l]["margin"] > threshold and dataset_gpt4[r]["margin"] <= threshold:
                    new_dataset.append({
                        "idx": idx,
                        "usrmsg": dataset_gpt3_5[l]["usrmsg"],
                        "target_output": dataset_gpt3_5[l]["target_output"],
                        "gt_label": dataset_gpt3_5[l]["gt_label"],
                        "pred_label": dataset_gpt3_5[l]["pred_label"],
                        "margin": dataset_gpt3_5[l]["margin"]
                    })
                    idx += 1
                l += 1
                r += 1
        
    # data_dict = {}
    # for i in range(len(labels)):
    #     label_str = str(labels[i]["gpt3.5"]) + "_" + str(labels[i]["gpt4"])
    #     data_dict[label_str] = data_dict.get(label_str, 0) + 1
    print(f"num of single selection: {len(new_dataset)}")
    for data in add_dataset:
        new_dataset.append({
            "idx": idx,
            "usrmsg": data["usrmsg"],
            "target_output": data["new_feedback"],
            "gt_label": data["gt_label"],
            "pred_label": data["pred_label"],
            "margin": data["margin"]
        })
        idx += 1

    train_dataset = []
    for data in new_dataset:
        messages = [
            {'role': 'system', 'content': 'You are a helpful and fair assistant that assesses the quality of AI-generated responses. Be objective and detailed.'},
            {'role': 'user', 'content': data["usrmsg"]},
            {'role': 'assistant', 'content': data["target_output"]},
        ]
        train_dataset.append({
            'messages': messages,
        })


    with open(new_dataset_path, "w", encoding="utf-8") as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Dataset saved to {new_dataset_path}")
    print(f"Dataset size: {len(train_dataset)}")

        

# generate_by_gpt()