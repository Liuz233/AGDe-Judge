import json
import argparse

def convert_winner_field(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # model_a = item.pop('model_a', None)
        # model_b = item.pop('model_b', None)
        # item['A'] = model_a
        # item['B'] = model_b
        # item.pop('messages', None)  
        # item.pop('instruction', None)
        # instruction = item.pop('orig_instruction', None)  
        # item['instruction'] = instruction
        # output_1 = item.pop('orig_response_A', None)
        # output_2 = item.pop('orig_response_B', None)
        # item['response_A'] = output_1
        # item['response_B'] = output_2
        # item.pop('conv_metadata', None)  
        # item.pop('category_tag', None)  

        winner = item.pop('winner', None) 
        label = winner
        # if winner == 'model_a':
        #     label = 'A'
        # elif winner == 'model_b':
        #     label = 'B'
        # else:
        #     label = 'TIE'
        item['label'] = label

        # winner = item.pop('label', None)  
        # if winner == 1:
        #     label = 'A'
        # elif winner == 2:
        #     label = 'B'
        # else:
        #     label = 'tie'
        # item['label'] = label

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    convert_winner_field(args.input, args.output)
