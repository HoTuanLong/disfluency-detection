import json

if __name__ == '__main__':
    with open("/home/ubuntu/long.ht/disfluency/datasets/word-level/test/seq.in", "r") as file1:
        input = [line.strip().split(" ") for line in file1]
    with open("/home/ubuntu/long.ht/disfluency/datasets/word-level/test/seq.out", "r") as file2:
        output = [line.strip().split(" ") for line in file2]
    
    for i in range(len(output)):
        if (len(input[i]) != len(output[i])):
            print("Error: ", input[i], "-", output[i])
        
        dict = {"words": input[i], "tags": output[i]}
        # json_object = json.dumps(dict, indent=2)
        with open("/home/ubuntu/long.ht/disfluency/datasets/word-level/test/test.json", "a", encoding='utf-8') as json_file:
            # print(dict)
            json.dump(dict, json_file, ensure_ascii=False)
            json_file.write('\n')